import sys
import os
import json
import numpy as np
import copy, time
import random
import requests
import traceback
from config import cfg_priv, merge_priv_cfg_from_file
from fishEye_lib import FishEye
from colormap import colormap
from ut import find_rect, cross_line, recorder, async_call

cfg_file = os.path.join(os.path.split(__file__)[0], 'face_analysis_config.yaml')
merge_priv_cfg_from_file(cfg_file)

N = 100


class FaceCounts(object):
    def __init__(self):
        super().__init__()
        self.FishEye = FishEye()

        self.tracks = dict()
        # 记录输入数组，以便于debug
        self.recorder = recorder(N)
        # 是否画图，可视化来debug
        self.debug = False
        self.II = 0  # 可视化时，保存用的名字自增名称
        self.visual_id = 0  # 可视化是否交叉时，保存用的名字自增名称
        self.curID = 0  # 帧号
        self.set_MACetc = False  # 判断是否有设定mac，进出门店的位置标识线等
        self.sendInfoList = []
        self.checkMark = {}  # 检查标记为mark的是否应该给个确定的身份：进出路过
        # 发送的基本信息格式send
        # self.content = {'media_id': -1, 'media_mac': "", "count_area_id": -1, "count_area_type": -1,
        #                 "in_num": 0, "out_num": 0, "pass_num": 0, "event_time": 0}
        # self.width = 1920
        # self.in_num = 0
        # self.out_num = 0
        # self.pass_num = 0
        # self.ratio = 0
        # self.out_info = {'list_track': [], 'list_box': [], 'list_id': [], 'solid': [], 'dotted': []}
        # assert cfg_priv.BUSS.COUNT.ENTRANCE_LINE['default'] != "None", \
        #     "cfg_priv.BUSS.COUNT.ENTRANCE_LINE['default'] != None"
        # assert len(cfg_priv.BUSS.COUNT.ENTRANCE_LINE['default']) != 0, \
        #     "len(cfg_priv.BUSS.COUNT.ENTRANCE_LINE['default']) != 0"
        # assert cfg_priv.OTHER.COUNT_MODE == True, \
        #     "cfg_priv.OTHER.COUNT_MODE == True"
        # self.redefine = False
        # self.rect = [[0, 0], [1, 1]]
        # self.entrance_line = [[2, 2], [4, 4]]

    def dummpy(self):
        '''发送统计消息'''
        if len(self.sendInfoList) > 0 and not self.debug:
            # for content in self.sendInfoList:
            #     self.oneItem(content)
            self.sendInfoList = []
        '''查看是否更新图框；如有更新'''
        pth = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'set.json')
        if os.path.exists(pth):
            print('set params is true')
            self.set_MACetc = True
            with open(pth, 'r')as f:
                params = json.load(f)
            self.media_id = params['media_id']
            self.media_mac = params['media_mac']
            self.media_rtsp = params['media_rtsp']
            self.record = params['debug']

            self.shopID = params['BUSS.COUNT.ROI_AREA_ID']
            self.lineType = params['BUSS.COUNT.ROI_AREA_TYPE']
            areas = []
            for t, keys in enumerate(['BUSS.COUNT.ROI_SOLID_LINE_AREA', 'BUSS.COUNT.ROI_DOTEED_LINE_AREA']):
                Items = []
                for i, shopid in enumerate(self.shopID):
                    if shopid not in params[keys]:
                        print('%s should equals to the shop IDS %s:' % (str(shopid), str(self.shopID)))
                        continue
                    Points = params[keys][shopid]
                    # todo: may change order here: rows, cols
                    points = np.array([[int(float(x) * 2.5), int(float(y) * 2.5)] for x, y in Points], dtype=np.int32)
                    Items.append(points)
                areas.append(Items)
            self.areas = list(zip(*areas))
            if self.record:
                new_pth = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'D_set.json')
                os.system('mv %s %s' % (pth, new_pth))
            else:
                os.system('rm %s' % pth)

    def canvas(self):
        img = np.ones((self.H, self.W + 300, 3), dtype=np.uint8) * 255
        img[:, :2] = 0
        img[:, -2:] = 0
        img[:2, :] = 0
        img[-2:, :] = 0
        if self.set_MACetc:
            for ii, (solid, dotted) in enumerate(self.areas):
                shop = self.shopID[ii]
                cur_id = self.lineType[shop]
                for pi in range(len(solid)):
                    cv2.line(img, (solid[(pi + 1) % len(solid)][0], solid[(pi + 1) % len(solid)][1]),
                             (solid[pi][0], solid[pi][1]), (80, 80, 80), 4)
                for pi in range(len(dotted)):
                    cv2.line(img, (dotted[(pi + 1) % len(dotted)][0], dotted[(pi + 1) % len(dotted)][1]),
                             (dotted[pi][0], dotted[pi][1]), (100, 100, 100), 2)
                cv2.putText(img, "T:%sS:%s" % (cur_id, shop), (solid[0][0] + 1, solid[0][1] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 80, 80), 2)
                cv2.putText(img, "T:%sS:%s" % (cur_id, shop), (dotted[0][0] + 1, dotted[0][1] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
        cv2.putText(img, "cur id-%d" % self.curID, (int(self.W * 0.6), int(self.H * 0.1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (80, 80, 80), 2)

        track = self.out_info['list_track']
        boxes = self.out_info['list_box']
        draw_id = self.out_info['list_id']
        colors = self.out_info['list_color']
        solid = self.out_info['solid']
        dotted = self.out_info['dotted']
        for ii in range(len(track)):
            points = track[ii]
            color = colors[ii]
            cur_id = draw_id[ii]
            line_s, line_d = solid[ii], dotted[ii]
            text = False
            for pi in range(1, len(points)):
                if abs(points[pi][0] - points[(pi - 1) % len(points)][0]) > self.W // 2: continue
                r_, g_, b_ = cl = color
                if line_s[pi] != '': cl = (255 - r_, 255 - g_, 255 - b_)
                if line_d[pi] != '': cl = ((355 - r_) % 200, (355 - r_) % 200, (355 - r_) % 200)
                cv2.line(img, (points[pi - 1][0], points[pi - 1][1]), (points[pi][0], points[pi][1]), cl, 3)
                text = True
            if text:
                cv2.putText(img, "id: %d" % cur_id, (points[0][0] + 1, points[0][1] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
            cv2.circle(img, (points[- 1][0], points[- 1][1]), 10, color, -1)

        self.II += 1
        cv2.imwrite('vs/base_demo%d.jpg' % self.II, img)

    def visual_check_intersection(self, aax, aay, bbx, bby, ccx, ccy, ddx, ddy):
        minx = min([aax, bbx, ccx, ddx]) - 10
        miny = min([aay, bby, ccy, ddy]) - 10
        maxx = max([aax, bbx, ccx, ddx]) - 10
        maxy = max([aay, bby, ccy, ddy]) - 10
        img = np.ones((maxy - miny + 30, maxx - minx + 30, 3), dtype=np.uint8) * 255
        img[:, :2] = 0
        img[:, -2:] = 0
        img[:2, :] = 0
        img[-2:, :] = 0
        cv2.line(img, (int(aax - minx), int(aay - miny)), (int(bbx - minx), int(bby - miny)), (0, 0, 0), 1)
        cv2.line(img, (int(ccx - minx), int(ccy - miny)), (int(ddx - minx), int(ddy - miny)), (0, 0, 0), 1)
        self.visual_id += 1
        cv2.imwrite('vs/check%d.jpg' % self.visual_id, img)

    def deter_in_out(self, aax, aay, bbx, bby):
        status = []
        if not self.set_MACetc:
            return status
        # 先实后虚，防止迈了一大步
        for i, (solid, dotted) in enumerate(self.areas):
            shop = self.shopID[i]
            for ii in range(len(solid)):
                ccx, ccy = solid[ii]
                ddx, ddy = solid[(ii + 1) % len(solid)]
                cross = cross_line(aax, aay, bbx, bby, ccx, ccy, ddx, ddy)
                if cross:
                    # if self.debug: self.visual_check_intersection(aax, aay, bbx, bby, ccx, ccy, ddx, ddy)
                    status.append(['s', shop])

            for ii in range(len(dotted)):
                ccx, ccy = dotted[ii]
                ddx, ddy = dotted[(ii + 1) % len(dotted)]
                cross = cross_line(aax, aay, bbx, bby, ccx, ccy, ddx, ddy)
                if cross:
                    # if self.debug: self.visual_check_intersection(aax, aay, bbx, bby, ccx, ccy, ddx, ddy)
                    status.append(['d', shop])
        return status

    def get_tracks(self, img_data, current_id, max_lost_frames=5):
        if self.set_MACetc:
            self.statics_in = dict.fromkeys(self.shopID, 0)
            self.statics_out = dict.fromkeys(self.shopID, 0)
            self.statics_passby = dict.fromkeys(self.shopID, 0)
        up_and_down_frames = 2
        for person in img_data["annotations"]:
            track_id = person["tracking_id"]
            global_id = person["global_id"]
            box_xywh = person["head_bbox"]
            position = [int(box_xywh[0] + box_xywh[2] / 2), int(box_xywh[1] + box_xywh[3] / 2)]

            if track_id in self.tracks:
                self.tracks[track_id]['boxes'] = box_xywh
                self.tracks[track_id]['track'].append(position)
                self.tracks[track_id]['latest_frame'] = current_id
                self.tracks[track_id]['solid'].append('')
                self.tracks[track_id]['dotted'].append('')
            else:
                self.tracks[track_id] = dict()
                self.tracks[track_id]['boxes'] = box_xywh
                self.tracks[track_id]['track'] = []
                self.tracks[track_id]['track'].append(position)
                self.tracks[track_id]['draw_id'] = global_id
                self.tracks[track_id]['status'] = True
                self.tracks[track_id]['draw'] = True
                self.tracks[track_id]['start_frame'] = current_id
                self.tracks[track_id]['latest_frame'] = current_id
                self.tracks[track_id]['solid'] = ['']
                self.tracks[track_id]['dotted'] = ['']
            if len(self.tracks[track_id]['track']) > 1:
                aax, aay = self.tracks[track_id]['track'][-2]
                bbx, bby = position

                # 检查是否标记为mark的可以认为是路过
                for c_id in list(self.checkMark.keys()):
                    c_key_s, c_key_d, c_shop = self.checkMark[c_id]
                    try:
                        c_record = len(self.tracks[c_id]['solid'])
                        if c_record - self.tracks[c_id][c_key_s] > 5:
                            self.tracks[c_id][c_key_d] = -1
                            self.tracks[c_id][c_key_s] = -1
                            self.statics_passby[c_shop] += 1
                            self.checkMark.pop(c_id)
                    except Exception as e:
                        self.checkMark.pop(c_id)
                        continue
                # if self.curID == 837:
                #     print()
                # todo: cross line
                rets = [] if abs(aax - bbx) > self.W // 2 else self.deter_in_out(aax, aay, bbx, bby)

                '''计数逻辑：
                首先，判断是否碰线，碰的什么样的线；
                其次，分析碰线次序，来判断是出还是进。
                    口口型1.碰一次实线后，至少碰一次虚线后：进；反之，出；碰两次实线，一段时间没有记录或一段时间没碰虚线，路过
                    回字型2.碰一次实线后，碰一次虚线：进；反之，出；碰两次实线，路过。
                    口字型3.将碰线视为：只有路过；或只有进出。
                最后，对于情况1，需对疑是路过进行标记，最终记做路过或进店（店口徘徊）
                TODO：未来，可添加位移量。目前只考虑碰线这一事件变量。
                所以，目前使用单个整数数字来表示碰线次序，浮点数来标记。单个数字即可完成。
                以上，为value值的意义和表示方法。
                key包含有shop信息，线类型信息。
                故目前使用key-value对来表示关系，key-value属于不同的id字典。
                '''
                for ret in rets:
                    types, shop = ret
                    id_record = len(self.tracks[track_id]['solid'])
                    old_key_S, old_key_D, new_key_S, new_key_D = (-1,) * 4
                    idx_key_s = '_'.join([str(shop), 's'])
                    idx_key_d = '_'.join([str(shop), 'd'])

                    if idx_key_s in self.tracks[track_id]:
                        old_key_S = self.tracks[track_id][idx_key_s]
                    if idx_key_d in self.tracks[track_id]:
                        old_key_D = self.tracks[track_id][idx_key_d]

                    # 如果经过一段时间仍没有消掉mark，或者有更新记录，那么就计入passby；删除实线框记录
                    if isinstance(old_key_D, float):  # 不管此id是碰了实线还是虚线，反正是碰线进来的了
                        if types == 'd':
                            self.statics_in[shop] += 1
                            self.tracks[track_id][idx_key_s] = -1
                        if types == 's':
                            self.statics_passby[shop] += 1
                            self.tracks[track_id][idx_key_s] = id_record
                        # 只有类型1才有mark，所以只考虑它就好了
                        self.tracks[track_id][idx_key_d] = -1
                        self.checkMark.pop(track_id)
                        continue
                    # 只有类型1才会出现，出店事件发生后，可能置身实线框内，又折回碰到虚线
                    if isinstance(old_key_S, float):
                        if types == 's':  # 就是出店了
                            old_key_S = -1
                        if types == 'd':  # 实锤置身实线框内
                            old_key_S = id_record - up_and_down_frames - 1

                    # 更新记录以及排除密集碰线的情况.暂时不管密集撞线情况
                    if types == 's':
                        self.tracks[track_id]['solid'][-1] = shop
                        # if id_record - old_key_S < up_and_down_frames and id_record >= 2: old_key_S = -1
                        self.tracks[track_id][idx_key_s] = id_record
                        new_key_S = id_record
                    if types == 'd':
                        self.tracks[track_id]['dotted'][-1] = shop
                        # if id_record - old_key_D < up_and_down_frames and id_record >= 2: old_key_D = -1
                        self.tracks[track_id][idx_key_d] = id_record
                        new_key_D = id_record

                    # 让new始终作为最新记录
                    if new_key_S == -1:
                        old_key_S, new_key_S = new_key_S, old_key_S
                    # 类型3会保留旧记录
                    if new_key_D == -1 and self.lineType[shop] != 3:
                        old_key_D, new_key_D = new_key_D, old_key_D
                    # 一步穿两线的情况，尤其对于类型1更常见
                    if new_key_D == new_key_S:
                        if old_key_D > 0:  # 穿过了一次虚线了
                            new_key_S += 1
                        if track_id in self.checkMark and self.checkMark[track_id][0] == idx_key_s:
                            new_key_D += 1
                            self.checkMark.pop(track_id)

                    # 用记录来计数
                    door_status = self._count(old_key_S, old_key_D, new_key_S, new_key_D, shop, self.lineType[shop])

                    # 成功计数一次之后，就需要更新记录了
                    if door_status == 'mark':  # 留意虚线框
                        self.tracks[track_id][idx_key_d] = -0.5
                        self.checkMark[track_id] = (idx_key_s, idx_key_d, shop)
                    if door_status == 'in':  # 删除实线框记录
                        # t1 2
                        self.tracks[track_id][idx_key_s] = -1
                        if self.lineType[shop] != 3:  # 类型3需要它监控事件出
                            self.tracks[track_id][idx_key_d] = -1
                    if door_status == 'out':  # 删除虚线框记录
                        # t1 2
                        self.tracks[track_id][idx_key_s] = -1
                        if self.lineType[shop] == 1:
                            self.tracks[track_id][idx_key_s] = -0.5  # 可能在实线框里，等待出框
                        self.tracks[track_id][idx_key_d] = -1

        for track_id in self.tracks.keys():
            if current_id - self.tracks[track_id]['latest_frame'] > max_lost_frames:
                img_data["delete_tracking_id"].append(track_id)
            if track_id in img_data["delete_tracking_id"]:
                self.tracks[track_id]['status'] = False

    def smart_judge(self, track_id):
        # 现有的碰线逻辑，不在需要连接线段了，先暂时废弃此函数
        '''拼接的基本逻辑是断点的判断；
        但是，目前使用的计数逻辑不是可逆向追溯，所以断点重连，对计数没有帮助。
        但是可以想办法，利用区间计算是否构成进出店等；但有多次计数的风险；
        也可以在删除本计数点时，断言进出店情况。
        '''
        # can a stop point of a line being a start point of another line?
        # 拼接断点相近，且位置相近的实例
        dead_id = self.tracks[track_id]
        up_and_down_frames = 2

        for cur_key, instance in self.tracks.items():
            if cur_key == track_id: continue
            if 0 <= instance['start_frame'] - dead_id['latest_frame'] < 2 * up_and_down_frames:
                if not (len(instance['track']) or len(dead_id['track'])): continue
                dis = np.array(instance['track'][0]) - np.array(dead_id['track'][-1])
                if np.sum(np.abs(dis) < 10) == 2:
                    # connect them
                    dead_id['track'].extend(instance['track'])
                    dead_id['solid'].extend(instance['solid'])
                    dead_id['dotted'].extend(instance['dotted'])
                    instance['track'] = dead_id['track']
                    instance['solid'] = dead_id['solid']
                    instance['dotted'] = dead_id['dotted']
                    instance['start_frame'] = dead_id['start_frame']
                    self.tracks.pop(track_id)
                    return True
        return False

    def count_num(self):
        tracks_tmp_up = copy.deepcopy(self.tracks)
        track_ids = tracks_tmp_up.keys()
        pop_ids = []
        for track_id in track_ids:
            occur = (self.curID == self.tracks[track_id]['latest_frame'])
            if self.tracks[track_id]['status']:
                if cfg_priv.OTHER.COUNT_DRAW:
                    if len(self.tracks[track_id]['track']) > 0 and occur:
                        self.draw_track(self.tracks[track_id])
            else:
                # is_combine = self.smart_judge(track_id)
                # if is_combine: continue
                if cfg_priv.OTHER.COUNT_DRAW:
                    if len(self.tracks[track_id]['track']) > 0 and occur:
                        self.draw_track(self.tracks[track_id])
                pop_ids.append(track_id)
                if track_id in self.checkMark:
                    self.checkMark.pop(track_id)

        for track_id in pop_ids:
            self.tracks.pop(track_id)

    # 计数逻辑的核心代码
    def _count(self, old_key_S, old_key_D, new_key_S, new_key_D, shop, lineType):
        '''注意：passby比较麻烦，因为经过两次实现框，且短时间内不经过虚线
        注意2：一旦计数成功，需要清空旧数据，以防虚假判断
        可以不用考虑必须经过两次实线框或两次虚线框，然后再碰虚实才能判读进出
        。why，经过了实线框，又经过虚线框，肯定是经过了进门线
        '''
        double_solid = False  # 两次碰触实线
        if old_key_S > 0 and new_key_S > 0 and old_key_D <= 0 and new_key_D <= 0:
            double_solid = True
        if lineType == 3:  # 只管虚线
            if old_key_D > 0 and new_key_D > 0:
                self.statics_out[shop] += 1
                return 'out'
            elif new_key_D > 0:
                self.statics_in[shop] += 1
                return 'in'
        else:
            # 先实后虚，则为进
            if new_key_S > 0 and new_key_D > 0 and new_key_D >= new_key_S:
                self.statics_in[shop] += 1
                return 'in'
            if lineType == 2:
                # 先虚后实，则为出
                if new_key_D > 0 and new_key_S > 0 and new_key_D < new_key_S:
                    self.statics_out[shop] += 1
                    return 'out'
                if double_solid:
                    self.statics_passby[shop] += 1
                    return 'out'  # 直接清除
            if lineType == 1:
                # 先虚后实实，则为出
                if new_key_D > 0 and new_key_S > 0 and old_key_S > 0 and new_key_D < new_key_S:
                    self.statics_out[shop] += 1
                    return 'out'
                if double_solid:
                    # mark and fcous on dot line
                    return 'mark'

    def draw_track(self, track):
        color_map = colormap()
        color = int(color_map[track['draw_id'] % 79][0]), int(color_map[track['draw_id'] % 79][1]), int(
            color_map[track['draw_id'] % 79][2])

        if track['draw']:
            if cfg_priv.OTHER.COUNT_DRAW_LESS:
                new_track = track['track'][-cfg_priv.OTHER.DRAW_TRACK_NUM:]
                solid = track['solid'][-cfg_priv.OTHER.DRAW_TRACK_NUM:]
                dotted = track['dotted'][-cfg_priv.OTHER.DRAW_TRACK_NUM:]
            else:
                new_track = track['track']
                solid = track['solid']
                dotted = track['dotted']
            if len(new_track) > 5:
                var_thred = 3
                var_x, var_y = np.var(np.array(new_track), axis=(0))
                # 如果想当长时间不动就不在显示了
                if var_x < var_thred and var_y < var_thred:
                    return
            self.out_info['solid'].append(solid)
            self.out_info['dotted'].append(dotted)
            self.out_info['list_track'].append(new_track)
            self.out_info['list_box'].append(track['boxes'])
            self.out_info['list_id'].append(track['draw_id'])
            self.out_info['list_color'].append(color)

    @async_call
    def oneItem(self, content):
        requests.post(url='http://172.16.104.247:5000/flow/pvcount', data=content)

    def send(self):
        if not self.set_MACetc:
            return
        in_num, out_num, pass_num = 0, 0, 0
        for shop in self.shopID:
            try:
                _in_num, _out_num, _pass_num = self.statics_in[shop], self.statics_out[shop], self.statics_passby[
                    shop]
            except Exception as e:
                _in_num, _out_num, _pass_num = (0,) * 3
                print(e)
            content = {'media_id': self.media_id, 'media_mac': self.media_mac, "count_area_id": shop,
                       "count_area_type": self.lineType[shop], "in_num": _in_num,
                       "out_num": _out_num, "pass_num": _pass_num,
                       "event_time": int(time.time())}
            in_num += content["in_num"]
            out_num += content['out_num']
            pass_num += content['pass_num']
            if content["in_num"] or content['out_num'] or content['pass_num']:
                self.sendInfoList.extend([content])
                if self.debug:
                    #self.canvas()
                    print(self.curID, content)

        self.out_info.update({'entran': in_num, 'pass_by': pass_num, 'out_num': out_num})

    # def doubleCheckSetParams(self):
    #     if not self.set_MACetc:
    #         new_pth = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'D_set.json')
    #         _pth = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'set.json')
    #         if os.path.exists(new_pth):
    #             os.system("cp %s %s" % (new_pth, _pth))

    def __call__(self, scores, boxes, classes, ratio_h, ratio_w, H, W):
        self.H, self.W = H, W
        self.curID += 1
        # 除跟踪线、人体框、人体id以及相对应的颜色外，额外的数据：
        # 进、出、路过
        self.out_info = {'list_track': [], 'list_box': [], 'list_id': [], 'list_color': [],
                         'entran': 0, 'pass_by': 0, 'out_num': 0, 'solid': [], 'dotted': [],
                         'rec': [[0, 0], [1, 1]], 'entrance_line': [[2, 2], [4, 4]]}
        save_input = self.set_MACetc and self.record
        save_input = True
        if save_input:
            C_scores, C_boxes, C_classes = scores.copy(), boxes.copy(), classes.copy()
        status = False
        # self.doubleCheckSetParams()#有毒
        # if self.curID == 416:
        #     print()
        try:
            ##################################
            self.dummpy()
            print(scores.shape)
            single_img_info_dict = self.FishEye(scores, boxes, classes, ratio_h, ratio_w, H, W)
            self.get_tracks(single_img_info_dict, self.curID)
            if len(single_img_info_dict['annotations']):
                status = True
            self.count_num()
            self.send()
            ##################################
        except Exception as e:
            traceback.print_exc(e)
            print(e)
        if save_input and status:
            self.recorder.save(C_scores, C_boxes, C_classes)

        return self.out_info


def sendall():
    p = [
        {'media_id': 36, 'count_area_id': '1', 'event_time': int(time.time()), 'count_area_type': 1, 'out_num': 1,
         'in_num': 0, 'media_mac': '00-02-D1-83-83-6E', 'pass_num': 0},
        {'media_id': 36, 'count_area_id': '1', 'event_time': int(time.time()), 'count_area_type': 1, 'out_num': 0,
         'in_num': 1, 'media_mac': '00-02-D1-83-83-6E', 'pass_num': 0},
        {'media_id': 36, 'count_area_id': '1', 'event_time': int(time.time()), 'count_area_type': 1, 'out_num': 0,
         'in_num': 1, 'media_mac': '00-02-D1-83-83-6E', 'pass_num': 0},

        {'media_id': 36, 'count_area_id': '1', 'event_time': int(time.time()), 'count_area_type': 1, 'out_num': 1,
         'in_num': 0, 'media_mac': '00-02-D1-83-83-6E', 'pass_num': 0},
        {'media_id': 36, 'count_area_id': '1', 'event_time': int(time.time()), 'count_area_type': 1, 'out_num': 0,
         'in_num': 1, 'media_mac': '00-02-D1-83-83-6E', 'pass_num': 0},
        {'media_id': 36, 'count_area_id': '1', 'event_time': int(time.time()), 'count_area_type': 1, 'out_num': 0,
         'in_num': 1, 'media_mac': '00-02-D1-83-83-6E', 'pass_num': 0}
    ]
    l = len(p)
    print('oooo')
    for i in range(100):
        content = p[i % l]
        requests.post(url='http://172.16.104.247:5000/flow/pvcount', data=content)
        print('.', i)


if __name__ == '__main__':
    from runProject import test_set

    test_set()
    # try:
    #     os.system("cp /srv/fisheye_prj/AI_Server/xxx_* /home/user/project/run_retina/build/")
    # except Exception as e:
    #     print("cp xxx* error")
    # try:
    #     os.system("cp /srv/fisheye_prj/AI_Server/utils/py_extension/D_set.json /home/user/project/run_retina/py_extension/set.json")
    # except Exception as e:
    #     print("cp set.json")
    # 进出店参数框
    # if os.path.exists('/home/user/project/run_retina/py_extension/D_set.json'):
    #     os.system("cp /home/user/project/run_retina/py_extension/D_set.json /home/user/project/run_retina/py_extension/set.json")
    # debug可视图
    try:
        os.system("rm /home/user/project/run_retina/py_extension/vs/*")
    except Exception as e:
        print("rm debug photo")

    fc = FaceCounts()
    fc.debug = True
    npz = '../build/xxx_%d.npz'
    n, N = 100, 100
    while True:
        if not os.path.exists(npz % n): break  # or n / 100 >= 10
        print(npz % n)
        data = np.load(npz % n)
        n += N
        data.allow_pickle = True
        scores, classes, boxes = data['s'], data['c'], data['b']
        # print(fc.set_MACetc)
        for s, c, b in zip(scores, classes, boxes):
            res = fc(s, b, c, 1.5, 1.6, 1920, 2880)
