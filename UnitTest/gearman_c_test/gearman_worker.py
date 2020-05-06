import requests
import json
import time
import sys
import traceback
import cv2
import requests
from urllib.parse import quote
import gearman
import base64
import numpy as np

gm_worker = gearman.GearmanWorker(['127.0.0.1:4730'])


def task_listener_reverse(gearman_worker, gearman_job):
    print("receve data")
    s=gearman_job.data

    img_s = json.loads(s)

    d64 = base64.b64decode(img_s['imgs'])
    nparr = np.fromstring(d64, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite('./imgs/woker_py.jpg', image)

    # 2 维度数组
    bbox = img_s['bbox']
    print(bbox)
    print("write img done")

    return "1"


if __name__ == '__main__':
    #fileName = generate_path()
    #upload_path(fileName)
    gm_worker.set_client_id('dph')
    gm_worker.register_task('dph', task_listener_reverse)
    print('start worker')
    gm_worker.work()
