import gearman
import json
import numpy as np
import os
import time

def client(detect_type, filePs, seg_param, segpx=0, segopt=-1, od_conf=0):
    """
    :param detect_type: 1: detect, 2: seg 3: detect + seg
    :param filePs:
    :param segpx: 0: 不返回 seg 坐标, 1: 返回坐标
    :param segopt: -1 不返回覆盖比, 0: 覆盖部分覆盖比 1: 裸露部分覆盖比
    :param od_conf: 0 检测阈值, 0 时用默认阈值, 百分比整数
    :return:
    """
    print('seg_param -- {}'.format(seg_param))
    gm_client = gearman.GearmanClient(['127.0.0.1:4730'])
    # completed_job_request = gm_client.submit_job("test_function", json.dumps(obj=dict(path=filePs)))
    completed_job_request = gm_client.submit_job("ls_test", filePs)

    # print(completed_job_request.result)
    return completed_job_request.result


def clent_person():
    gm_client = gearman.GearmanClient(['127.0.0.1:4730'])
    persons = []
    person = dict(name="None", rec=False, id=-1,
                confidence=-1, timestamp=int(time.time()*1000))
    persons.append(person)
    json_str = json.dumps(dict(persons=persons))
    print(json_str)
    completed_job_request = gm_client.submit_job("DPH_TRACKER_SERVER", json_str, background=True)
    return completed_job_request.result


if __name__ == '__main__':
    time_start = time.time()
    img_path = '/home/xinxueshi/workspace/tensorrt/capi/data/ssd/P000006.png'
    # result = client(detect_type=2, filePs=img_path, seg_param=[[100, 200, 300, 400], [500, 600, 700, 800]], segpx=1,
    #                 segopt=0)
    result = clent_person()
    print('****')
    print(result)
    print('****')
    # print(os.path.exists(result))
    # print(np.load(result))
    print(time.time()-time_start)
    # print(np.load('/home/user/workspace/priv-0220/privision_test/P000014.npy'))
