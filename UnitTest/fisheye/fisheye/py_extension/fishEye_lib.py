import sys, os
import numpy as np
from ut import Profiler
from airport_untities import npbbox_iou

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
# from fh_tracking.fhtracker import HFtracker
from box_tracking import HFtracker


# from torch_extension.pre_count.core.config import cfg_priv
# from torch_extension.pre_count.core.config import merge_priv_cfg_from_file
# cfg_file = 'torch_extension/face_analysis_config.yaml'
# print('cfg file:', cfg_file)
# merge_priv_cfg_from_file(cfg_file)
# import time
# import cv2
# import argparse
# from multiprocessing import Process
# from torch_extension.pre_count.face_lib import FaceLib
# # from source.modules.face_3dparsing import pad_ldmk_square_box, preprocess
# # from source.modules.face_detection import SSDPersonFaceDet as PersonFaceDet
# from torch_extension.retnet import *


#
# parser = argparse.ArgumentParser(description='dewarp fisheye')
# parser.add_argument('--size', metavar='height width', type=int, nargs='+',
#                     help='input size (square) or sizes (h w) to use when generating TensorRT engine',
#                     default=[640, 1280])
# parser.add_argument('--batch', metavar='batch', type=int, default=1)


class FishEye(object):
    def __init__(self):
        super().__init__()
        # self.FaceDet, self.lib_info = get_lib(engine_file)
        self.tracker = HFtracker()
        self.profiler = Profiler([])
        self.id_viewer = {}
        self.boundary_area = 10
        self.pixel_sz = 16  # fisheye dewarp
        self.overlap = True
        # this is used for tracking off the wall of two sides
        self.shift = 500 if self.overlap else 0
        self.nms_thres = 0.45
        self.score_thres = 0.15
        self.del_box = False
        self.Evaluate = False
        self.target_cls = 1
        # self.split_map = {}

        self.IDs = 0

        #######

        self.id_mapping = {}
        self.split_parts = 3
        self.body_class_id = 0
        # self.boundary_area = 0.10  # total ratio is 10%, both side is 5%
        # assert 0 < self.boundary_area < 0.2, 'padding ratio for left & right'
        # self.mode = 2
        self.up_ids, self.dn_ids = [], []

    def tracking(self, detection_rects, track_w, track_h, cut_lines):
        '''
        aim to pack the tracking function of facelib here
        '''
        # delete_tracking_id = []
        # tracking_id = list(range(len(detection_rects)))
        tracking_id, delete_tracking_id = \
            self.tracker.tracking_Frame_Hungarian(detection_rects, track_w, track_h, cut_lines)
        out = {'tracking_id': tracking_id,
               'delete_tracking_id': delete_tracking_id}
        return out

    def gap_detect_overlap(self, ori_box, score, thres=0.45, Is_del_box=False):
        # get rid of some box using NMS among overlap area
        # build connection using clone boxes as well as overlap area boxes
        # 1.find out no overlap mask
        # 'pad': pad, 's1': [W // 3 * 2, H // 2], 's2': [W // 3, H], 's3': [W, H]}
        self.split_map = {}
        mask = 1
        if len(ori_box) == 0: return
        # offset = 200
        # s1, s2, ss2, s3, pad = self.gt_gap['s1'], self.gt_gap['s2'], self.gt_gap['ss2'], \
        #                        self.gt_gap['s3'], self.gt_gap['pad']
        # x1, x2, y2 = ori_box[:, 0], ori_box[:, 2], ori_box[:, 3]
        # sp1 = (y2 < s1[1]) * (x2 > s1[0] - offset) * (x1 < s1[0] + pad + offset)
        # # sp22 = (y2 < ss2[1]) * (x2 > ss2[0] - offset) * (x1 < ss2[0] + pad + offset)
        # sp2 = (y2 > s2[1]) * (x2 > s2[0] - offset) * (x1 < s2[0] + pad + offset)
        # sp3 = (y2 > s3[1]) * (x2 > s3[0] - offset) * (x1 < s3[0] + pad + offset)
        # mask = sp1 + sp2 + sp3
        # if not mask.sum(): return

        nms = npbbox_iou(ori_box.copy(), ori_box.copy())

        nms -= np.eye(len(ori_box))
        ratio_mask = np.zeros_like(nms)
        try:
            ratio_mask[np.arange(len(ori_box)), np.argmax(nms, axis=0)] = 1
            # ratio_mask *= (nms > thres)
        except Exception as e:
            print(e)
        del_ids = []
        for p, s in np.argwhere(ratio_mask * (nms > thres) * mask):
            if not Is_del_box:
                # 不删除box，防止误删错误。用关系构建的方式。
                self.split_map[p] = s
            else:
                # 删除box，忽略误删情况。降低跟踪时多个框的id会发生混乱的可能性
                if score[p] > score[s]:
                    del_ids.append(s)
                else:
                    del_ids.append(p)
        ori_box[list(set(del_ids))] *= 0

    def mapping(self, dete_res, tracking_id):
        id_mapping, track_ids, delete_ids = {}, [], []
        ori_box_len = dete_res['ori']
        pre_len = len(dete_res['upleft_idx']) + ori_box_len
        for i, idx in enumerate(dete_res['upleft_idx']):
            id_mapping[tracking_id[idx]] = tracking_id[ori_box_len + i]
            id_mapping[tracking_id[ori_box_len + i]] = tracking_id[idx]
        for i, idx in enumerate(dete_res['upright_idx']):
            id_mapping[tracking_id[idx]] = tracking_id[pre_len + i]
            id_mapping[tracking_id[pre_len + i]] = tracking_id[idx]
        # conections between ids of boxes using overlap or split line
        # only for new occured boxes
        split_map, sp_del = {}, []
        try:
            for p, s in self.split_map.items():
                # if p == 'del':  continue
                split_map[tracking_id[p]] = tracking_id[s]
                split_map[tracking_id[s]] = tracking_id[p]
        except Exception as e:
            print(e)
        return id_mapping, split_map

    def gap_detect_overlap2(self, ori_box, score, thres=0.45, Is_del_box=False):
        # get rid of some box using NMS among overlap area
        # build connection using clone boxes as well as overlap area boxes

        self.split_map = {}
        mapping = {}
        boxes = ori_box['box'].copy()
        extras = ori_box['extra'].copy()
        ul = ori_box['ul'].copy()
        base_score, extra_score, ul_score = score

        def keepCheck(box, extra, b_score, e_score):
            nms = npbbox_iou(box, extra)  # IOU
            # check those have same height box
            m = np.array([0, 1, 0, 1, 0], dtype=np.float32)
            nms_zone = npbbox_iou(box * m, extra * m)
            # 1.delete boxes of same person
            nms_idx = np.argmax(nms, axis=0)
            nms_score = np.max(nms, axis=0) > thres
            # a.those have high IOU; or b. those has less IOU but with same height
            nms_con = nms_score + ((nms > thres / 2) * (nms_zone > 0.8))[nms_idx, np.arange(len(nms_idx))]

            # replace by high score. but may try place -- which far from split line
            # keey extra_idx for that we want keep only one
            # extra_idx = nms_con * (box[nms_idx, 4] < extra[:, 4])  # distance
            extra_idx = nms_con * (b_score[nms_idx] < e_score)
            box_idx = nms_idx[extra_idx]

            box[box_idx] = extra[extra_idx]
            b_score[box_idx] = e_score[extra_idx]
            return box_idx, extra_idx

        if len(extra_score):
            box_idx, extra_idx = keepCheck(boxes, extras, base_score, extra_score)
            for p, s in zip(box_idx, np.arange(len(extra_idx))[extra_idx] + len(boxes)):
                mapping[p] = s
        if len(ul_score) and len(extra_score):
            extra_idx1, ul_idx = keepCheck(extras, ul, extra_score, ul_score)
            box_idx = ori_box['ul_idx'][ul_idx]
            for p, s in zip(box_idx, extra_idx1 + len(boxes)):
                mapping[p] = s

        # 全部的base；部分的extra
        if len(extra_score):
            box = np.vstack([boxes, extras])  # [~extra_idx]
            score = np.hstack([base_score, extra_score])  # [~extra_idx]
        else:
            box, score = boxes, base_score
        ret = {'box': box[:, :4], 'score': score, 'W': ori_box['W'], 'H': ori_box['H'],
               'map': mapping, 'cut': ori_box['cut']}

        # ul_start = ori_box['ori']
        # ul_con = (nms_idx >= ul_start) * extra_idx
        # ul_idx = nms_idx[ul_con] - ul_start
        # box_ids = ori_box['map'][ul_idx]
        # # replace box of left head
        # box[box_ids] = np.clip(extra[ul_con] - ori_box['ul'], a_min=0, a_max=ori_box['ul'][0])
        # b_score[box_ids] = e_score[ul_con]
        #
        # # 2. add useful box and mapping to left head
        # extra_ids = ul_start + np.arange(ul_con.sum())
        # ul_con += ori_box['jump']
        # ori_box['box'] = np.vstack([box[:ul_start], extra[ul_con]])
        # ori_box['score'] = np.hstack([b_score[:ul_start], e_score[ul_con]])
        #
        # for p, s in zip(box_ids, extra_ids):
        #     mapping[p] = s
        # ori_box['map'] = mapping
        return ret

    def gap_detect(self, ori_box):
        # send some box clone to certain loction.
        # build connection between some boxes using clone ones
        self.gap_box = {}
        box = ori_box.copy()
        n, b4 = box.shape
        self.split_map = {}
        if n == 0: return
        assert b4 == 4, 'box shape should be n*4 here!'
        gt_gap, bW, bH = self.gt_gap['gap'], self.gt_gap['splitW'], self.gt_gap['splitH']
        minus = np.abs(box - gt_gap)
        # 1. near gap line
        mask = np.min(minus, axis=0) < self.boundary_area
        mask = mask[:, 2] + mask[:, 0]
        # iou of y axis
        box[(box[:, 2] > bW) * (box[:, 3] > bH)] -= gt_gap[3, 0, :]
        boxes = box.reshape(-1, 1, 4)
        b1, b2, b3 = boxes.shape
        one = np.ones((b1, b1, b3), dtype=np.float32)
        bb = np.concatenate((boxes * one, box * one), axis=-1)
        # yy3=np.hstack((boxes[:,:,:]*one,box[:,:]*one))
        yy_top_inter = np.max(bb[:, :, [1, 1 + b3]], axis=-1)
        yy_bot_inter = np.min(bb[:, :, [3, 3 + b3]], axis=-1)
        yy_inter = yy_bot_inter - yy_top_inter  # nn
        yy_union = ((bb[:, :, 3] - bb[:, :, 1]) + (bb[:, :, 7] - bb[:, :, 5]) - yy_inter)
        yy_iou = yy_inter / yy_union
        yy_ = (1 - (yy_iou - np.eye(b1))) < 0.4  # near 1

        # 2. x<y when merge two boxes -> is square?
        de_xx_iou = np.abs((bb[:, :, 0] - bb[:, :, 6])) / yy_union
        union_xx_yy_ratio = np.max(np.stack([de_xx_iou.T, de_xx_iou]), axis=0)
        union_xx_yy_ratio += (np.eye(b1) + (yy_inter < 0)) * 5
        ratio_mask = np.zeros_like(union_xx_yy_ratio)
        try:
            ratio_mask[np.arange(b1), np.argmin(union_xx_yy_ratio, axis=0)] = 1
            ratio_mask *= (union_xx_yy_ratio < 1.2)
        except Exception as e:
            print(e)
        # 3. distance lt 10
        xx = np.abs(bb[:, :, 0] - bb[:, :, 6]) < (2 * self.boundary_area)  # nn

        # condition
        # when merge 2 boxes, it is also square(iou of y near 1 and xx/yy lt 1(?));
        # 2 boxes near split line;
        # two boxes are close enough
        condition = np.argwhere(mask * yy_ * xx * ratio_mask)
        # old_key = set(list(self.split_map.keys()))
        # new_key = ['del']
        for p, s in condition:
            self.split_map[p] = s
            self.split_map[s] = p
            # new_key.extend([p, s])

        # del_key = list(old_key - set(new_key))
        # for p in del_key:
        #     self.split_map.pop(p)
        # self.split_map['del'] = del_key

    def fliter(self, boxes):
        # 1: w/h<1.2
        wh_thred = 1.2
        w = boxes[..., 2] - boxes[..., 0]
        h = boxes[..., 3] - boxes[..., 1] + 1e-5
        wh_ratio = w / h
        wh_idx = wh_ratio > wh_thred
        return wh_idx

    def rearrange_box_sp3(self, scores, boxes, classes, w, h, ratio, base_batch=3):
        assert len(boxes) % 3 == 0, 'boxes should be 3*n'
        boxes[..., 0::2] *= ratio[0]
        boxes[..., 1::2] *= ratio[1]
        n, nb, _ = boxes.shape
        # 过滤部分数据
        idx_ = self.fliter(boxes)
        scores[idx_] *= 0.5
        # x1,y1,x2,y2; 4. distance; 5.overlaped marker; span marker; 6. left most or right most marker;
        new_boxes = np.concatenate((boxes.astype(np.float32), np.zeros((n, nb, 3), dtype=np.float32)), axis=-1)
        new_boxes[new_boxes[:, :, 2] >= w * 2 // 3, 5] = 1  # 4. those overlaped
        ww = w * 2 // 3
        pad = int((ratio[0] - ratio[1]) * ww)
        start_x_3 = ww * 2
        start_y_3 = 0  # h // 2  # ==0 尽可能的减少死id，也就是尽量不用映射id的方式了
        start_x_2 = ww
        start_y_2 = 0  # h//2# ==0 尽可能的减少死id，也就是尽量不用映射id的方式了
        # Span over stop line: ww
        jump = (new_boxes[..., :, 0] < ww) * (new_boxes[..., :, 2] > ww)
        new_boxes[jump, 5] = 2
        # distance
        distance = np.min(np.stack([np.abs(new_boxes[..., :, 0:3:2] - ww),
                                    new_boxes[..., :, 0:3:2]], 2).reshape(*boxes.shape[:2], -1), -1)
        new_boxes[..., 4] = distance
        # mark left
        left_m = (new_boxes[0::3, :, 0] <= ww // 4)
        new_boxes[0::3][left_m, 6] = 1
        left_shift_start = np.array([ww * 3, 0, ww * 3, 0], dtype=np.float32)  # 相对位移，没有shift
        # mark right
        right_m = (new_boxes[2::3, :, 2] >= 3 * ww // 4)
        new_boxes[2::3][right_m, 6] = 2
        right_shift_start = np.array([-ww * 3, 0, -ww * 3, 0], dtype=np.float32)

        # list on a line
        new_boxes[2::3, :, 0:3:2] += start_x_3
        new_boxes[2::3, :, 1:4:2] += start_y_3
        new_boxes[1::3, :, 0:3:2] += start_x_2
        new_boxes[1::3, :, 1:4:2] += start_y_2
        idx = (scores >= self.score_thres) * (classes == self.target_cls)
        # shiftting in case of offing the left side wall
        new_boxes[..., 0:3:2] += self.shift
        new_W, new_H = (ww * 3 + ww // 4 + 2 * self.shift), h // 2
        # cut_lines
        cut_lines = np.array([0, ww, ww * 2, ww * 3, ww + pad, ww * 2 + pad, ww * 3 + pad])
        cut_lines += self.shift

        # split to batch
        res_box, res_score = [], []
        for i in range(len(boxes) // 3):
            tmp_idx = idx[i * 3:(i + 1) * 3]
            res_box.append(new_boxes[i * 3:(i + 1) * 3][tmp_idx])
            res_score.append(scores[i * 3:(i + 1) * 3][tmp_idx])

        def extra_arrange(box, score):
            box = box[:, :4]
            upleft_idx = box[:, 6] == 1
            upleft = box[upleft_idx] + left_shift_start
            upright_idx = box[:, 6] == 2
            upright = box[upright_idx] + right_shift_start
            info = {'upleft_idx': upleft_idx, 'upright_idx': upright_idx, 'ori': len(box),
                    'box': np.vstack([box, upleft, upright]),
                    'W': new_W, 'H': new_H, 'cut': cut_lines}
            ss = np.hstack([score, score[upleft_idx], score[upright_idx]])
            return info, ss

        def extra_arrange_overlap(box, score):
            '''
            the main idea: 图片切分成重叠部分和其他；
            重叠部分和其他部分计算IOU，高者替换低者；
            鱼眼图片的最左\最右和其他三条切割线略有不同，所以单独拿出来处理
            '''
            idx = box[:, 5] == 1  # overlaped box
            jump = box[:, 5] == 2  # span box, has interaction with idx
            upleft_idx = box[:, 6] == 1
            extra_idx = jump + idx  # both overlap and span # (~jump) * idx  # overlap but without span

            upleft = box[upleft_idx, :5] + np.hstack([left_shift_start, 0])
            extra = box[extra_idx, :5]  # manual overlaped box
            res_box = box[~extra_idx, :5]
            info = {'extra': extra, 'box': res_box, 'ul': upleft, 'W': new_W, 'H': new_H,
                    'ul_idx': np.where(upleft_idx[~extra_idx])[0], 'cut': cut_lines}
            ss = score[~extra_idx], score[extra_idx], score[upleft_idx].copy()  # same as box & extra & ul

            return info, ss

        if self.overlap:
            res = [extra_arrange_overlap(box, score) for box, score in zip(res_box, res_score)]
        else:
            res = [extra_arrange(box, score) for box, score in zip(res_box, res_score)]
        return res

    def set_global_id(self, track_res, dete_res, id_viewer):
        tracking_id, delete_tracking_id = track_res['tracking_id'], track_res['delete_tracking_id']
        if 'id_pool' not in id_viewer:
            id_pool = list()
        else:
            id_pool = list(id_viewer['id_pool'])

        def get_unique_id(exist_id):
            old_set = set(exist_id)
            new_set = list(range(len(old_set) + 1))
            return list(set(new_set) - old_set)[0]

        def set_globlId(up_id):
            down_id = id_mapping[up_id] if up_id in id_mapping else None
            sp_id = split_map[up_id] if up_id in split_map else \
                split_map[down_id] if down_id in split_map else None
            # get maxmium one as it's id
            global_ids = []
            up_union_ids = []
            down_union_ids = []
            if up_id in id_viewer:
                local_global_id = id_viewer[up_id]['global_id']
                global_ids.append([up_id, local_global_id])
                up_union_ids = id_viewer[up_id]['mapping_id']
                for ids in up_union_ids:
                    if ids not in id_viewer: continue  # todo
                    local_global_id = id_viewer[ids]['global_id']
                    global_ids.append([ids, local_global_id])
            if down_id in id_viewer:
                down_union_ids = id_viewer[down_id]['mapping_id']
                if down_id not in up_union_ids:
                    local_global_id = id_viewer[down_id]['global_id']
                    global_ids.append([down_id, local_global_id])

            # if sp_id is not None: down_union_ids.append(sp_id)
            # IOU generate spid & up_id; but they may not refer to same person
            # only when up_id or sp_id is fresh
            # if sp_id not in id_viewer and sp_id not in up_union_ids:
            #     local_global_id = id_viewer[sp_id]['global_id']
            #     global_ids.append([sp_id, local_global_id])
            #     up_union_ids.append(sp_id)
            # down_union_ids.append(sp_id)
            # sp_union_ids = id_viewer[sp_id]['mapping_id']
            # sp_union_ids.append(up_id)
            # if down_id is not None: sp_union_ids.append(down_id)

            # get glocal id
            if len(global_ids) > 0:  # vote
                ids = np.array(global_ids)
                local_global_id = np.argmax(np.bincount(ids[:, 1]))
            else:  # random
                local_global_id = get_unique_id(id_pool)

            # add alias for up id
            if down_id is not None:
                up_union_ids.append(down_id)
                down_union_ids.append(up_id)
                id_viewer[down_id] = {'global_id': local_global_id, 'mapping_id': list(set(down_union_ids))}
            if sp_id is not None and sp_id not in id_viewer:  # absolutly fresh
                union_ids = [up_id] if down_id is None else [up_id, down_id]
                id_viewer[sp_id] = {'global_id': local_global_id, 'mapping_id': union_ids}
            id_viewer[up_id] = {'global_id': local_global_id, 'mapping_id': list(set(up_union_ids))}
            id_pool.append(local_global_id)
            return local_global_id

        def del_global_ids(del_ids, rm_item=True):
            for ids in del_ids:
                if ids not in id_viewer: continue
                mapping_ids = id_viewer[ids]['mapping_id']
                g_id = id_viewer[ids]['global_id']
                do_nothing = True
                # 与之相关者移除
                for mapping_id in mapping_ids:
                    if mapping_id in id_viewer:
                        # 找到他关联项的关联内容
                        maps = id_viewer[mapping_id]['mapping_id']
                        if ids in maps:
                            maps.pop(maps.index(ids))  # del it from its friends
                            # id_viewer[mapping_id]['mapping_id'] = maps  # 好像多余
                            # 移除死亡id
                            if id_viewer[mapping_id]['global_id'] == g_id:
                                do_nothing = False
                # 移除idmapping关联项
                # if ids in id_mapping:
                #     # print('*'*10)
                #     target = id_mapping[ids]
                #     id_mapping.pop(ids)
                #     id_mapping.pop(target)
                if do_nothing:
                    delete_ids.append(g_id)
                if rm_item: id_viewer.pop(ids)  # del it

        # prepare connection ids. for all instance
        id_mapping, delete_ids = {}, []
        split_map, sp_del = {}, []
        if self.overlap:
            for p, s in dete_res['map'].items():
                # if p == 'del':  continue
                id_mapping[tracking_id[p]] = tracking_id[s]
                id_mapping[tracking_id[s]] = tracking_id[p]
        else:
            id_mapping, split_map = self.mapping(dete_res, tracking_id)

        # main process---------------
        def distribut_global_id():
            # 找到跟踪id和全局分配的id间映射关系。
            temp_ids = []
            for ids in tracking_id:  # [:ori_box_len]
                gid = set_globlId(ids)
                temp_ids.append(gid)
            return temp_ids

        track_ids = distribut_global_id()
        if len(set(track_ids)) != len(track_ids):
            bad_ids = []
            # 需要处理冲突 npbbox_iou
            temp_map, temp_idx = {}, {}
            ious = npbbox_iou(dete_res['box'], dete_res['box'])
            for k_id, (temp_id, real_id) in enumerate(zip(track_ids, tracking_id)):
                if temp_id in temp_map:
                    temp_map[temp_id].append(real_id)
                    temp_idx[temp_id].append(k_id)
                else:
                    temp_map[temp_id] = [real_id]
                    temp_idx[temp_id] = [k_id]
                if len(temp_map[temp_id]) > 1:
                    _k, _j = temp_idx[temp_id]
                    iou = ious[_k, _j]
                    if iou < self.nms_thres:
                        # 移除后来的
                        bad_ids.append(max(temp_map[temp_id]))
                    temp_map[temp_id].pop(-1)
                    temp_idx[temp_id].pop(-1)
            if len(bad_ids):
                delete_tracking_id.extend(list(set(bad_ids)))
                # for all deleted id
                del_global_ids(delete_tracking_id)
                track_ids = distribut_global_id()

        id_viewer['id_pool'] = set(id_pool)
        return track_ids, delete_ids

    def x1y1x2y2_x1y1wh(self, box):
        x1, y1, x2, y2 = box
        w, h = x2 - x1 + 1, y2 - y1 + 1
        return np.array([x1, y1, w, h])

    def engine2(self, scores, boxes, classes, ratio_h, ratio_w, H, W):
        for i in range(len(boxes) // 3):
            yield ratio_h, ratio_w, H, W

    def engine(self, scores, boxes, classes, ratio_h, ratio_w, H, W):
        ratio = (ratio_w, ratio_h)
        # split line 3 -> x1,y1,h = H//2
        gt_gap = np.array([[0, 0], [W // 3 * 2, 0], [W // 3, H // 2], [W, H // 2]])
        gt_gap = np.tile(gt_gap, 2).reshape(-1, 1, 4)
        self.batch = batch = 1
        batch_size = c = 3
        # if self.IDs >= 69:
        #     print()
        self.profiler.start('post_')
        # arrange boxes to a line & mapping to a dict
        all_detes_boxes = self.rearrange_box_sp3(scores, boxes, classes, W, H, ratio)
        results = []
        # new_W, new_H = int(W // 3 * 2 * (3 + 0.1)), H
        for ii, (detes_res, score) in enumerate(all_detes_boxes):

            detes_boxes = detes_res['box']
            info = {"delete_tracking_id": [], "annotations": []}
            if (len(detes_boxes)):
                if not self.overlap:
                    self.gap_detect(detes_boxes[:detes_res['ori']])
                else:
                    # replace those with high IOU
                    detes_res = self.gap_detect_overlap2(detes_res, score, self.nms_thres, Is_del_box=self.del_box)
                if not self.Evaluate:
                    ###todo:
                    track_id = self.tracking(detes_res['box'].copy(), detes_res['W'], detes_res['H'], detes_res['cut'])
                    tracking_id, delete_tracking_id = self.set_global_id(track_id, detes_res, self.id_viewer)
                    detes_boxes_ = detes_res['box']

                else:
                    tracking_id, delete_tracking_id = detes_res['score'], []  # score
                    detes_boxes_ = detes_res['box']  # box
                # print(detes_boxes_)
                anno = []
                detes_boxes_[..., 0:3:2] -= self.shift
                for id, box in zip(tracking_id, detes_boxes_):
                    x1, y1, x2, y2 = box
                    if x2 > W:
                        if x1 < W - 1:
                            box[2] = W - 1
                        xx1, x2 = max(0, x1 - W), x2 - W
                        y1 += H // 2
                        y2 += H // 2
                        tmp = np.asarray([xx1, y1, x2, y2])
                        if x1 > W - 1:
                            box[:] = tmp[:]
                        else:
                            anno.append({"tracking_id": id, "global_id": id, "head_bbox": self.x1y1x2y2_x1y1wh(tmp)})
                            # bbbb.append(tmp)
                    anno.append({"tracking_id": id, "global_id": id, "head_bbox": self.x1y1x2y2_x1y1wh(box)})  # xywh
                #     bbbb.append(box)
                # print(np.array(bbbb))
                # get info of images
                info = {"delete_tracking_id": delete_tracking_id, "annotations": anno}

                self.profiler.stop('post_')

            yield info

    def __call__(self, scores, boxes, classes, ratio_h, ratio_w, H, W):
        for res in self.engine(scores, boxes, classes, ratio_h, ratio_w, H, W):
            self.IDs += 1
            return res

    def show_eclipse(self):
        self.profiler.bump('infer')
        # print(self.profiler.totals['infer'])
        msg = ''
        if (self.profiler.totals['infer'] > 10):
            for name in self.profiler.names:
                if not name: continue
                msg += '({}: {:.3f}s) '.format(name, self.profiler.means[name])
            msg += ', {:.1f} im/s'.format(self.batch / self.profiler.means['infer'])
            print(msg, flush=True)
            self.profiler.reset()
