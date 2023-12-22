import numpy as np

class Global_Id_Identificator():
    
    def __init__(self):
        self.global_id_cnt = 1        
        self.id_table = np.array([[1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 0, -1]]) # global id, local id1, local id2, bbox1, bbox2, hits, match state
        self.obj_nan  = np.array([1e-1, 1e-1, 1e-1, 1e-1])
        
        self.frame_idx = 0
        self.prev_id_tables_complete = False
        self.prev_id_memory_n = 3 # cnt
        self.prev_id_tables = [[]] * self.prev_id_memory_n
        
        
    def Check_2D_ID_Table(self, local_id1, local_id2):
        '''
        1) 입력 (local_id1, local_id2)가 ID Table에 id1 혹은 id2 한개라도 겹친다면 겹치는 행의 인덱스를 반환        
        2) 겹치지 않는다면 -1을 반환
        
        [참고] id는 반드시 겹치지 않거나 한가지만 겹치기 때문에 id1과 id2가 모두 겹친다면 ID Table고 겹치는 인덱스(id1, id2) 둘 중 아무거나 반환
        '''
        
        state1 = np.where(self.id_table[:,1] == local_id1)[0]
        state2 = np.where(self.id_table[:,2] == local_id2)[0]
        
        # 반드시 state는 1개이거나 없어야함.
        assert len(state1) <= 1
        assert len(state2) <= 1
        
        if len(state1) == 1:
            return state1[0]
        elif len(state2) == 1:
            return state2[0]
        else:
            return -1
    
    def Check_1D_ID_Table(self, local_id, dimension):
        '''
        1) 입력 (local_id)가 ID Table의 입력 dimension에 겹친다면 겹치는 행의 인덱스를 반환
        2) 겹치지 않는다면 -1을 반환
        '''
        assert dimension == 1 or dimension == 2
        
        state = np.where(self.id_table[:,dimension] == local_id)[0]
        
        if len(state) == 1:
            return state[0]
        else:
            return -1
    
    def Push_ID_Table(self, local_id1, local_id2, bbox_cam1, bbox_cam2):
        '''
        입력 (local_id1, local_id2)을 Global ID에 제일 뒤에 추가
        실제로 Global ID를 나타내는 global_id_cnt를 +1
        매칭완료 상태를 True로 변경
        '''
        (x1, y1, x2, y2), (x3, y3, x4, y4) = bbox_cam1, bbox_cam2
        push = np.array([self.global_id_cnt, local_id1, local_id2, x1, y1, x2, y2, x3, y3, x4, y4, 1, 1])
        self.id_table = np.vstack((self.id_table, push))
        self.global_id_cnt += 1
     
    def Update_ID_Table(self, Global_Idex, local_id1, local_id2, bbox_cam1, bbox_cam2):
        '''
        입력 (Global_Idex, local_id1, local_id2)를 이용하여
        ID Table 수정
        
        [추가]
        local_id1, local_id2가 모두 부여되었을 때, (즉, 누구도 -1이 아닐 때)
        이전 id table의 id가 매칭되어 있지 않다면, 0으로 초기화
        '''
        
        prev_id_table = self.prev_id_tables[0] # 이전 id_table
        state = 0
        if local_id1 != -1 and local_id2 != -1:
            condition1 = (np.all(prev_id_table[:, 1:3] == [int(local_id1), -1], axis=1)).sum()
            condition2 = (np.all(prev_id_table[:, 1:3] == [-1, int(local_id2)], axis=1)).sum()
            state = condition1 + condition2

        (x1, y1, x2, y2), (x3, y3, x4, y4) = bbox_cam1, bbox_cam2
        self.id_table[Global_Idex, 1] = local_id1
        self.id_table[Global_Idex, 2] = local_id2
        self.id_table[Global_Idex, 3:-2] = np.array([x1, y1, x2, y2, x3, y3, x4, y4])
        self.id_table[Global_Idex, -1] = 1
        
        if state != 0:
            self.id_table[Global_Idex, -2] = 0
        else:
            self.id_table[Global_Idex, -2] += 1
            
    def Delete_unmatched_t_ID_Table(self):
        '''
        매칭완료 상태 상태가 False인 인덱스를 찾아 삭제
        '''

        Global_Idex = np.where(self.id_table[:,-1] == -1)[0]
        self.id_table = np.delete(self.id_table, Global_Idex, axis=0)
    
    def prev_id_tables_update(self):
        '''
        prev_id_table에 id_table를 저장(최근 prev_id_memory_n 개수 만큼만)
        '''
        if self.frame_idx == self.prev_id_memory_n:#3
            self.prev_id_tables_complete = True

        # for i in range(self.prev_id_memory_n-1):# i = 0, 1
        #     self.prev_id_tables[self.prev_id_memory_n-i-1] = self.prev_id_tables[self.prev_id_memory_n-i-2]
        self.prev_id_tables[2] = self.prev_id_tables[1]
        self.prev_id_tables[1] = self.prev_id_tables[0]
        self.prev_id_tables[0] = self.id_table.copy()
        
        self.frame_idx += 1
        
    
    def update(self, matched, unmatched, matched_bbox, unmatched_bbox, id):        
        id1, id2 = id
        # k_{t-1}와 k_{t}의 매칭완료 상태를 모두 False로 초기화
        self.id_table[:, -1] = -1
        
        ## t 시점에서 t-1시점 비교
        # matched
        for (obj_cam1, obj_cam2), (bbox_cam1, bbox_cam2) in zip(matched, matched_bbox):
            local_id1 = id1[obj_cam1]
            local_id2 = id2[obj_cam2]
            
            case = self.Check_2D_ID_Table(local_id1, local_id2)
            # 생성 조건        
            if case == -1:
                self.Push_ID_Table(local_id1, local_id2, bbox_cam1, bbox_cam2)
                
            # 수정 조건
            else:
                self.Update_ID_Table(case, local_id1, local_id2, bbox_cam1, bbox_cam2)
        
        # unmatched
        for obj_cam1, bbox_cam1 in zip(unmatched[0], unmatched_bbox[0]):
            local_id1 = id1[obj_cam1]
            case = self.Check_1D_ID_Table(local_id1, dimension=1)
            # 생성 조건
            if case == -1:
                self.Push_ID_Table(local_id1, -1, bbox_cam1, self.obj_nan)

            # 수정 조건
            else:
                self.Update_ID_Table(case, local_id1, -1, bbox_cam1, self.obj_nan)


        for obj_cam2, bbox_cam2 in zip(unmatched[1], unmatched_bbox[1]):
            local_id2 = id2[obj_cam2]
            case = self.Check_1D_ID_Table(local_id2, dimension=2)         
            # 생성 조건
            if case == -1:
                self.Push_ID_Table(-1, local_id2, self.obj_nan, bbox_cam2)

            # 수정 조건
            else:
                self.Update_ID_Table(case, -1, local_id2, self.obj_nan, bbox_cam2)

        # t-1 시점에서 t시점 비교
        self.Delete_unmatched_t_ID_Table()