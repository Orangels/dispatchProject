import gearman
import cv2
import json
import sys
import time
import gearman
import traceback
from face_engine import FaceRecognise
from utils.config_utils import *
import numpy as np
import base64

face_reco = 0
# persons feature databaser
y = yaml_config()

threshold_default = 0.7
gm_worker = gearman.GearmanWorker(['localhost:4730'])


def task_listener_reverse(gearman_worker, gearman_job):
    """
    :param gearman_worker:
    :param gearman_job:
    :mode  0 quere person,  1 get feature , Add person
    :return:
    """
    global face_reco
    global y

    try:
        # parse b64 img
        s = gearman_job.data
        img_s = json.loads(s)
        bboxes = map(tuple, img_s['bbox'])
        mode = 0

        img_start = time.time()

        d64 = base64.b64decode(img_s['imgs'])
        nparr = np.fromstring(d64, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_name = './static/upload/{}.jpg'.format(int(time.time()))
        cv2.imwrite(img_name, image)
        img_end = time.time()

        if mode == 0:
            # pose reid
            img_ori_fea_arr = []
            personDB = y.config['personDB']
            persons = []

            start = time.time()
            for bbox in bboxes:
                infer_start = time.time()
                img_ori_fea = face_reco(image, bbox)
                _, img_ori_fea = face_reco.l2_norm(img_ori_fea)
                img_ori_fea_arr.append((img_ori_fea, bbox))
                infer_end = time.time()
                print("inference time cost : {}".format(infer_end-infer_start))

            for img_ori_fea in img_ori_fea_arr:
                for i in range(len(personDB['persons'])):
                    person = compare_persons(img_ori_fea[0][0], i)
                    person['bbox'] = img_ori_fea[1]
                    if person['confidence'] >= threshold_default:
                        persons.append(person)
                        break
            end = time.time()
            print('**********')
            print('img time cost %s' % str(img_end-img_start))
            print('**********')
            print('time cost %s' % str(end-start))
            print('**********')
            #return dic
            dic_json = dict(person=persons, infer_time=(end-start), img_time=(img_end-img_start))
            print(dic_json)
            return json.dumps(obj=dic_json)
        elif mode == 1:
            # face get feature
            start = time.time()
            fea_vec = ssdNet(img, show_box=False, show_kpts=False, face_compare=True, test=False, type=type)
            if fea_vec is not None:
                end = time.time()
                print('**********')
                print('img time cost %s' % str(img_end - img_start))
                print('**********')
                print('time cost %s' % str(end - start))
                print('**********')
                fea_str = ':'.join(map(str, fea_vec))
                return fea_str
            else:
                return 'error'
    except Exception as e:
            print(e)
            traceback.print_exc()
            return 'error'


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float16)
    vec2 = np.array(vec2, dtype=np.float16)

    len_vec1 = np.sqrt(vec1.dot(vec1))
    len_vec2 = np.sqrt(vec2.dot(vec2))

    if len_vec1 * len_vec2 == 0:
        return -1

    return vec1.dot(vec2) / (len_vec1 * len_vec2)


def add_persons(face_reco):
    global y
    personDB = y.config['personDB']

    for i in range(len(personDB['persons'])):
        if 'feature' in personDB['persons'][i].keys():
            continue
        img1 = cv2.imread(personDB['persons'][i]['img'])
        a = face_reco(img1, tuple(personDB['persons'][i]['bbox']))
        # print(a)
        _, A = face_reco.l2_norm(a)
        # print(A)
        personDB['persons'][i]['feature'] = A[0].tolist()
    y.config = dict(personDB=personDB)


def compare_persons(person_0_fea, person_1):
    personDB = y.config['personDB']
    person_1_fea = np.asanyarray(personDB['persons'][person_1]['feature'])
    compare_result = np.dot(person_0_fea, person_1_fea)
    print('result = {}'.format(compare_result))
    return dict(name=personDB['persons'][person_1]['name'], img=personDB['persons'][person_1]['img'],
                            bbox=personDB['persons'][person_1]['bbox'], confidence=compare_result)


def compare_persons_dbs(person_0, person_1):
    personDB = y.config['personDB']
    print(cosine_similarity(personDB['persons'][person_0]['feature'], personDB['persons'][person_1]['feature']))
    person_0_fea = np.asanyarray(personDB['persons'][person_0]['feature'])
    person_1_fea = np.asanyarray(personDB['persons'][person_1]['feature'])
    compare_result = np.dot(person_0_fea, person_1_fea)
    print(' {} vs {}  = {}'.format(person_0, person_1, compare_result))
    return compare_result
    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        face_reco = FaceRecognise()
    else:
        channel_id = int(sys.argv[1])
        face_reco = FaceRecognise(gpu_id=channel_id)
    print('worker start')
    gm_worker.set_client_id('python-worker')
    gm_worker.register_task('DHP_face', task_listener_reverse)
    gm_worker.work()

    # face_reco = FaceRecognise()
    # add_persons(face_reco)
    # compare_persons_dbs(0, 1)
