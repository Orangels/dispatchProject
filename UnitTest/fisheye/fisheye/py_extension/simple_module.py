import os
import numpy as np
from pre_count_lib import FaceCounts
from fishEye_lib import FishEye
from ToDB import connectDB
import traceback
from ut import async_call

ratio_w, ratio_h, H, W = (0,) * 4
FC = FaceCounts()
FE = FishEye()
fishObj = connectDB()


def simple_func(data, scale):
    print('*' * 10, data.shape, data.dtype, scale)
    return data


def set_param(r_h, r_w, HH, WW):
    global ratio_w, ratio_h, H, W
    ratio_h, ratio_w, H, W = r_h, r_w, HH, WW


def box_info(scores, classes, boxes, img, media_id, frame_id, mac):
    sendToDatabase(img, media_id, frame_id, mac)
    return get_info(scores, classes, boxes)


# @async_call
def get_info(scores, classes, boxes):
    # FC.debug = False
    try:
        # assert boxes.dtype == np.float32 and boxes.shape == (3, 100, 4)
        # assert scores.dtype == np.float32 and scores.shape == (3, 100)
        # assert classes.dtype == np.float32 and classes.shape == (3, 100)
        info = FC(scores, boxes, classes, ratio_h, ratio_w, H, W)
    except Exception as e:
        info = {'list_track': [], 'list_box': [], 'list_id': [], 'list_color': [],
                'entran': 0, 'pass_by': 0, 'out_num': 0, 'solid': [], 'dotted': [],
                'rec': [[0, 0], [1, 1]], 'entrance_line': [[2, 2], [4, 4]]}
        traceback.print_exc()

    # settle results
    list_id = np.array(info['list_id'], dtype=np.float32)
    list_id = list_id.clip(0)

    # list_track
    list_track, list_color, list_track_num = [], [], []
    for i, x in enumerate(info['list_track']):
        list_track_num.append(len(x))
        list_track.extend(x)
        list_color.extend(info['list_color'][i])
    list_track = np.array(list_track, dtype=np.float32)
    list_color = np.array(list_color, dtype=np.float32)
    list_track_num = np.array(list_track_num, dtype=np.float32)
    list_track = list_track.clip(0)
    list_color = list_color.clip(0)
    list_track_num = list_track_num.clip(0)
    # list_box
    list_box = np.array(info['list_box'], dtype=np.float32)
    list_box = list_box.clip(0)

    # support data
    static = np.array([info['entran'], info['pass_by'], info['out_num']], dtype=np.float32)
    static = static.clip(0)
    support = info['entrance_line']
    support.extend(info['rec'])
    support = np.array(support, dtype=np.float32)
    assert len(list_id) == len(list_track_num), 'len same'
    assert len(list_id) == len(list_box), 'len same 2'
    assert (3 * len(list_id)) == len(list_color), 'len same 3'
    assert len(list_track) == list_track_num.sum(), 'len same 4'
    return list_id.copy(), list_track.copy(), list_track_num.copy(), list_box.copy(), static, support, list_color
    # return list_id, list_track, list_track_num, list_box, static, support, list_color


@async_call
def sendToDatabase(img, media_id, frame_id, mac):
    fishObj.push_out(media_id, mac, frame_id, img)


if __name__ == '__main__':
    from runProject import test_set

    try:
        os.system("cp /srv/fisheye_prj/AI_Server/utils/py_extension/D_set.json "
                  "/home/user/project/run_retina/py_extension/set.json")
    except Exception as e:
        print("cp set.json")
    try:
        os.system("rm /home/user/project/run_retina/py_extension/vs/*")
    except Exception as e:
        print("rm debug photo")
    test_set(True)
    FC.debug = True
    npz = '../build/xxx_%d.npz'
    n = N = 100
    set_param(1.5, 1.6, 1920, 2880)
    while True:
        if not os.path.exists(npz % n) or n / 100 >= 10: break
        print(npz % n)
        data = np.load(npz % n)
        n += N
        data.allow_pickle = True
        scores, classes, boxes = data['s'], data['c'], data['b']
        print(FC.set_MACetc)
        for s, c, b in zip(scores, classes, boxes):
            res = box_info(s, c, b,0,0,0,0)
