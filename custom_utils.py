import cv2
import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

human_class = 0

font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 0, 0)

max_weights = 0.15
max_hits    = 15

def filter_human(dets, ):
    state = (dets[:, -1] == human_class)
    return dets[state, :]

def draw_rectangle(image, dets, cam_type):
    if cam_type == 1:
        margin = 0
    elif cam_type == 2:
        margin = 360
    
    for d in dets:
        x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
        cv2.rectangle(image, (x1, y1+margin), (x2, y2+margin), color, 2)

def draw_matching_line(image, assignments, trks, id):
    
    for assign in assignments:
        pos1 = trks[0][trks[0][:, -1] == id[0][assign[0]], :-1][0]
        pos2 = trks[1][trks[1][:, -1] == id[1][assign[1]], :-1][0]
        
        cx1, cy1 = int((pos1[0] + pos1[2])//2), int((pos1[1] + pos1[3])//2)
        cx2, cy2 = int((pos2[0] + pos2[2])//2), int((pos2[1] + pos2[3])//2)
        
        cv2.line(image, (cx1, cy1), (cx2, cy2+360), color, 2)

def draw_id(image, trks1, trks2, GII):
    for trk in trks1:
        _, _, x2, y2 = int(trk[:4][0]), int(trk[:4][1]), int(trk[:4][2]), int(trk[:4][3])
        local_id = int(trk[4])
        GIilLI1 = np.where(GII.id_table[:, 1] == local_id)[0][0]
        text = str(int(GII.id_table[GIilLI1, 0]))
        cv2.putText(image, text, (x2, y2), font, 1, color, 2)

    for trk in trks2:
        _, _, x2, y2 = int(trk[:4][0]), int(trk[:4][1]), int(trk[:4][2]), int(trk[:4][3])
        local_id = int(trk[4])
        GIilLI2 = np.where(GII.id_table[:, 2] == local_id)[0][0]
        text = str(int(GII.id_table[GIilLI2, 0]))
        cv2.putText(image, text, (x2, y2+360), font, 1, color, 2)


def calculate_non_assignments(cam1_obj_cnt, cam2_obj_cnt, assignments):
    
    A = np.arange(cam1_obj_cnt)
    B = np.arange(cam2_obj_cnt)
    
    for cam1_idx, cam2_idx in reversed(assignments):
        A = np.delete(A, np.where(A==cam1_idx)[0])
        B = np.delete(B, np.where(B==cam2_idx)[0])
        
    return [list(A), list(B)]

        
def extract_obj_image(image , trks):
    resize_dets_images = np.zeros((1, 256, 128, 3))
    
    for trk in trks:
        x1, y1, x2, y2 = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3])
        x1, x2, y1, y2 = max(0, x1), min(x2, 640-1), max(0, y1), min(y2, 360-1)
        
        resize_det_image = cv2.resize(image[y1:y2, x1:x2, :], (128, 256))
        resize_dets_images = np.concatenate((resize_dets_images, np.expand_dims(resize_det_image, axis=0)), axis=0)
    
    if len(resize_dets_images[1:,]) == 0 :
        return []
    else :
        return resize_dets_images[1:,]

def cosine_similarity(feature1, feature2):
    return 1 - cosine(feature1, feature2)

def calculate_cosine_similarity_matrix(features1, features2):
    num_people_cam1 = len(features1)
    num_people_cam2 = len(features2)
    
    similarity_matrix = np.zeros((num_people_cam1, num_people_cam2))
    
    for i in range(num_people_cam1):
        for j in range(num_people_cam2):
            similarity_matrix[i, j] = cosine_similarity(features1[i], features2[j])
    
    return similarity_matrix

def calculate_hungarian_algorithm(similarity_matrix):
    row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
    
    assignments = list(zip(row_indices, col_indices))
    cos_similarities = similarity_matrix[row_indices, col_indices]
    
    return np.array(assignments), cos_similarities

def Extract_bbox_info(matched, unmatched, trks1, trks2):
    matched_bbox = []
    for matched_i in matched:
        cam1_idx, cam2_idx = matched_i
        tmp = [list(trks1[cam1_idx, :4]), list(trks2[cam2_idx, :4])]
        matched_bbox.append(tmp)
    
    unmatched_bbox_cam1 = []
    for unmatched_i in unmatched[0]:    
        cam1_idx = unmatched_i
        tmp = list(trks1[cam1_idx, :4])
        unmatched_bbox_cam1.append(tmp)
    
    unmatched_bbox_cam2 = []
    for unmatched_i in unmatched[1]:    
        cam2_idx = unmatched_i
        tmp = list(trks2[cam2_idx, :4])
        unmatched_bbox_cam2.append(tmp)
        
    return np.array(matched_bbox), (np.array(unmatched_bbox_cam1), np.array(unmatched_bbox_cam2))

def calculate_weighted_similarity_matrix(similarity_matrix, trks1, trks2, GII):
    weighted_similarity_matrix = similarity_matrix.copy()
    
    if GII.prev_id_tables_complete:
        table_prev = GII.prev_id_tables[0] # t-1 시점의 id_table
        
        # [global id, cam1 id, cam2 id]
        id_info = table_prev[:, [0, 1, 2, -2]]
        prev_matched = id_info[~np.any(id_info[:, [1, 2]] == -1, axis=1)]
        
        prev_cam1_id = prev_matched[:, 1]
        prev_cam2_id = prev_matched[:, 2]
        prev_hits    = prev_matched[:, 3]
        
        weighted_rc = []
        
        for id1_i, id2_i, hits in zip(prev_cam1_id, prev_cam2_id, prev_hits):
            # 동일한 id가 있는 지 확인
            r = np.where(trks1[:,4] == id1_i)[0]
            c = np.where(trks2[:,4] == id2_i)[0]
            if len(r) == 1 and len(c) ==1 :
                weighted_rc.append([r[0], c[0], hits])

        for r, c, hits in weighted_rc:
            k = min(20, max_hits)
            weighted_similarity_matrix[r][c] += max_weights/max_hits*k

    return weighted_similarity_matrix