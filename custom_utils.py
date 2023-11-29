import json, cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 0, 0)

def read_json(path):
    with open(path, 'r') as f:
        json_dt = json.load(f)
    return json_dt

def draw_rectangle(image, dets, cam_type=1):
    if cam_type == 1:
        margin = 0
    elif cam_type == 2:
        margin = 360
    
    for d in dets:
        x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
        cv2.rectangle(image, (x1, y1+margin), (x2, y2+margin), (255, 0, 0), 2)
    
        
def convert_trks_shape(obj):
    '''
    아래와 같이 traker의 출력 형태로 변환
    [id, center_x, center_y, w, h] -> [x1, y1, x2, y2, id]
    (0 ~ 1, 0 ~ 1) -> (0 ~ 640, 0 ~ 360)
    '''
    obj = np.array(obj)
    obj = obj[obj[:, 6] == '0']
    
    xywh = obj[:,1:5].astype(np.float32)
    id   = obj[:,0:1].astype(np.float32)
    
    xyxy_copy = xywh.copy()
    id_copy   = id.copy()
    
    xyxy_copy[:, 0] = xywh[:,0] - xywh[:,2]/2
    xyxy_copy[:, 1] = xywh[:,1] - xywh[:,3]/2
    xyxy_copy[:, 2] = xywh[:,0] + xywh[:,2]/2
    xyxy_copy[:, 3] = xywh[:,1] + xywh[:,3]/2
    id_copy;    
    
    xyxy_copy[:,::2]  = xyxy_copy[:,::2] * 640
    xyxy_copy[:,1::2] = xyxy_copy[:,1::2] * 360
    
    return np.concatenate([xyxy_copy, id_copy], axis=1).astype(np.int32)

def calculate_similarity_matrix(trks1, trks2):
    '''
    이미 정답을 알고 있으므로(Global ID) 
    cosine similarity를 이용하여 유사도를 계산하지 않고,
    동일한 객체일 경우 0.9의 cost를 가지는 similarity_matrix 생성
    '''
    
    global_id1 = trks1[:, 4]
    global_id2 = trks2[:, 4]
        
    r = len(global_id1)
    c = len(global_id2)
    
    similarity_matrix = np.zeros((r, c)).astype(np.float32)
    
    for i in range(r):
        for j in range(c):
            similarity_matrix[i, j] = (global_id1[i] == global_id2[j])*0.9
    
    return similarity_matrix
    
def calculate_hungarian_algorithm(similarity_matrix):
    '''
    similarity_matrix를 이용하여 매칭
        assignment       : 매칭된 쌍
        cos_similarities : 매칭된 쌍의 cos_similarities
    '''
    
    row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
    
    assignments = list(zip(row_indices, col_indices))
    cos_similarities = similarity_matrix[row_indices, col_indices]
    
    return np.array(assignments), cos_similarities

def calculate_non_matched(trks1, trks2, assignments):
    '''
    각 cam에서 매칭되지 않은 id의 인덱스 계산
        A : cam1에서 매칭되지 않은 id의 인덱스
        B : cam2에서 매칭되지 않은 id의 인덱스
    '''
    
    A = np.arange(len(trks1))
    B = np.arange(len(trks2))
    
    for cam1_idx, cam2_idx in reversed(assignments):
        A = np.delete(A, np.where(A==cam1_idx)[0])
        B = np.delete(B, np.where(B==cam2_idx)[0])
    
    return [A, B]