import math
import sys, os
from Hungarian.Hungarian import HungarianAlgorithm


class HFtracker(object):
    def __init__(self):
        self.out_boundary_delete = True
        self.frames_numout_boundary_delete = 0
        self.current_max_id = 0
        self.img_w = 0
        self.img_h = 0
        self.cost_th = 1
        self.frames_num = 0

        self.tracking_infos = []
        self.max_mismatch_times = 30
        self.near_boundary_th = 0.05
        self.iou_cost_weight = 4

        self.size_cost_weight = 1
        self.ratio_cost_weight = 1
        self.mismatchTimes_cost_weight = 0.2
        self.boundary_cost_weight = 0
        self.max_distance = 1.0
        self.distance_cost_weight = 0.5 * self.max_distance
        self.max_cost = self.iou_cost_weight + self.distance_cost_weight + self.size_cost_weight + self.ratio_cost_weight + self.mismatchTimes_cost_weight + self.boundary_cost_weight

    def intersection_over_union(self, box1, box2):

        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min((box1.x + box1.width), (box2.x + box2.width))
        y2 = min((box1.y + box1.height), (box2.y + box2.height))
        over_area = max((x2 - x1), 0) * max((y2 - y1), 0)
        iou = float(over_area) / (box1.width * box1.height + box2.width * box2.height - over_area)
        return iou

    def near_boundary(self, box1):
        x2 = box1.x + box1.width
        y2 = box1.y + box1.height
        x1_boundary = float(box1.x) < min(50, self.img_w * self.near_boundary_th)
        y1_boundary = float(box1.y) < min(50, self.img_h * self.near_boundary_th)
        x2_boundary = float(self.img_w - x2) < min(50, self.img_w * self.near_boundary_th)
        y2_boundary = float(self.img_h - y2) < min(50, self.img_h * self.near_boundary_th)

        return x1_boundary or y1_boundary or x2_boundary or y2_boundary

    def cal_total_cost(self, box1, info):

        box2 = info.rect
        total_cost = 0

        s1 = box1.width * box1.height
        s2 = box2.width * box2.height
        size_cost = float(math.fabs(s1 - s2)) / (min(s1, s2))  # (0,++)

        xy_ratio1 = float(box1.width) / box1.height
        xy_ratio2 = float(box2.width) / box2.height
        ratio_cost = max(xy_ratio1, xy_ratio2) / min(xy_ratio1, xy_ratio2) - 1  # [0,++)

        center_x1 = box1.x + box1.width / 2
        center_y1 = box1.y + box1.height / 2
        center_x2 = box2.x + box2.width / 2
        center_y2 = box2.y + box2.height / 2
        distance_cost = math.sqrt((center_x1 - center_x2) * (center_x1 - center_x2) + (center_y1 - center_y2) * (
                center_y1 - center_y2))  # [0,++)
        distance_cost = distance_cost / math.sqrt(min(s1, s2)) / self.max_distance  # [0,++)

        mismatchTimes_cost = float(info.mismatch_times / self.max_mismatch_times)  # [0,1)

        # boundary_cost
        boundary_cost = float(self.near_boundary(box2))

        iou_cost = 1 - self.intersection_over_union(box1, box2)

        # cout<<"size_cost= "<<size_cost<<" ratio_cost= "<<ratio_cost<<" distence_cost= "<<distence_cost<<" iou_cost= "<<iou_cost<<endl

        if (size_cost > 2 or ratio_cost > 1.5 or distance_cost > 1.5):
            total_cost = self.max_cost
            return total_cost

        total_cost = iou_cost * self.iou_cost_weight + size_cost * self.size_cost_weight + ratio_cost * self.ratio_cost_weight + \
                     distance_cost * self.distance_cost_weight + mismatchTimes_cost * self.mismatchTimes_cost_weight + boundary_cost * self.boundary_cost_weight

        return total_cost

    def cal_costMatrix(self, detection_rects):
        total_cost_matrixs = []
        for i in range(len(detection_rects)):
            total_cost_matrix = []
            for j in range(len(self.tracking_infos)):
                # total_cost =1- intersection_over_union(detection_rects[i],self.tracking_infos[j].rect) # Iou COST
                total_cost = self.cal_total_cost(detection_rects[i], self.tracking_infos[j])
                total_cost_matrix.append(total_cost)

            total_cost_matrixs.append(total_cost_matrix)

        return total_cost_matrixs

    def match_tracking_detinfo(self, detection_rects):
        total_cost_matrixs = self.cal_costMatrix(detection_rects)
        HungAlgo = HungarianAlgorithm()
        mathch_rects = []
        for i in range(len(self.tracking_infos)):
            self.tracking_infos[i].match_now = False
        if (len(detection_rects)):
            assignment_tracking_id = HungAlgo.Solve(total_cost_matrixs, len(self.tracking_infos), len(detection_rects))
            # detection_rects.size() == assignment_tracking_id.size()
            for i in range(len(detection_rects)):
                tracking_id = assignment_tracking_id[i]
                mathch_rect = matches()
                mathch_rect.det_id = i
                if tracking_id >= 0:
                    mathch_rect.tracking_id = tracking_id
                    mathch_rect.iou = 1 - total_cost_matrixs[i][tracking_id]
                    mathch_rect.obj_id = self.tracking_infos[tracking_id].obj_id
                    # if(mathch_rect.iou < IoU_th)
                    if (total_cost_matrixs[i][tracking_id] >= self.max_cost * self.cost_th):
                        mathch_rect.obj_id = -1
                    else:
                        self.tracking_infos[tracking_id].match_now = True
                        self.tracking_infos[tracking_id].mismatch_times = 0
                else:
                    mathch_rect.obj_id = -1
                mathch_rects.append(mathch_rect)

        return mathch_rects

    def tracking_Frame_Hungarian(self, detection_rects_, img_w, img_h, cut_line):

        detection_rects = []
        if not detection_rects_ is None:
            detection_rects_[:, 2:4] -= detection_rects_[:, 0:2]
            for i in range(detection_rects_.shape[0]):
                rect = Rect()
                rect.x, rect.y, rect.width, rect.height = detection_rects_[i, 0:4]
                detection_rects.append(rect)
        self.img_h = img_h
        self.img_w = img_w
        tracking_result = []
        tracking_end_id = []
        # tracking
        if (self.frames_num == 0):
            for j in range(len(detection_rects)):
                track_info = tracker_info()
                track_info.rect = detection_rects[j]
                track_info.obj_id = j
                track_info.mismatch_times = 0
                track_info.iou_less = False
                self.tracking_infos.append(track_info)
                tracking_result.append(self.current_max_id)
                self.current_max_id += 1
                track_info.match_now = True
        else:
            match_rectinfos = self.match_tracking_detinfo(detection_rects)
            for j in range(len(match_rectinfos)):
                if (match_rectinfos[j].obj_id == -1):
                    rect = detection_rects[match_rectinfos[j].det_id]
                    self.current_max_id += 1
                    match_rectinfos[j].obj_id = self.current_max_id
                    track_info = tracker_info()
                    track_info.rect = rect
                    track_info.obj_id = self.current_max_id
                    track_info.mismatch_times = 0
                    track_info.iou_less = False
                    track_info.match_now = True
                    self.tracking_infos.append(track_info)

                else:
                    self.tracking_infos[match_rectinfos[j].tracking_id].rect = detection_rects[
                        match_rectinfos[j].det_id]
                # rectangle( image_input, self.tracking_infos[match_rectinfos[j].tracking_id].rect, Scalar( 0, 255, 255 ), 6, 8 )

            self.tracking_infos_size = len(self.tracking_infos)
            id = 0
            while id < self.tracking_infos_size:
                if (self.out_boundary_delete):
                    # 如果跟踪器出边缘，快时间内删除跟踪器
                    if (self.near_boundary(self.tracking_infos[id].rect) and self.tracking_infos[id].mismatch_times <= (
                            self.max_mismatch_times * 0.9 - 1)):
                        self.tracking_infos[id].mismatch_times = self.max_mismatch_times * 0.9 - 1
                if (self.tracking_infos[id].match_now):
                    self.tracking_infos[id].mismatch_times = 0
                else:
                    self.tracking_infos[id].mismatch_times += 1

                # 如果多次跟踪不到删除多余的跟踪器
                # 或多个跟踪器重叠较大，删除(合并)多余的跟踪器
                if not self.tracking_infos[id].iou_less:
                    for id_ in range(self.tracking_infos_size):
                        if self.intersection_over_union(self.tracking_infos[id].rect,
                                                        self.tracking_infos[id_].rect) > 0.1:
                            self.tracking_infos[id].iou_less = True
                            self.tracking_infos[id_].iou_less = True

                # is_less =(!self.tracking_infos[id].match_now && self.tracking_infos[id].iou_less)
                is_less = (self.tracking_infos[id].iou_less)
                is_time_out = self.tracking_infos[id].mismatch_times > self.max_mismatch_times
                # if(is_less | is_time_out)
                if (is_time_out):
                    # cout<< "erase Tracker: id = "<< self.tracking_infos[id].obj_id << endl
                    tracking_end_id.append(self.tracking_infos[id].obj_id)
                    self.tracking_infos.pop(id)
                    id -= 1
                    self.tracking_infos_size -= 1
                ##################################
                for line in cut_line:
                    left = self.tracking_infos[id].rect.x - line
                    right = line - self.tracking_infos[id].rect.x + self.tracking_infos[id].rect.width
                    near_cut = (left < 2 and left >= 0) and (right >= 0 and right < 2)
                    if (not is_time_out and near_cut and self.tracking_infos[id].mismatch_times > 2):
                        self.tracking_infos[id].mismatch_times += 3  # (n+1)x > 30
                ##################################
                id += 1
            # cout<<"max_id= "<<self.current_max_id<<"  tracker_num= "<<self.tracking_infos.size()<<endl
            for j in range(len(match_rectinfos)):
                tracking_result.append(match_rectinfos[j].obj_id)
        self.frames_num += 1
        return tracking_result, tracking_end_id


class Rect:
    pass


class matches:
    pass


class tracker_info:
    pass
