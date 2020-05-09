import gearman
import cv2
import json
import sys
import time
import gearman
import traceback
from face_engine import FaceRecognise
from utils.config_utils import *
from tools.utils import *
import numpy as np
import base64
import uuid

sys.path.append('./Pet-face-lib')
sys.path.append('./Pet-face-lib/tools')
from tools.face_lib import *


UPLOAD_FOLDER = 'static/uploads/'  # 保存文件位置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

upload_pic_url = 'http://127.0.0.1:5000/waring_img'

keypoint = 0
face_reco = 0
face_lib = 0
# persons feature databaser
y = yaml_config()

threshold_default = 0.5
gm_worker = gearman.GearmanWorker(['localhost:4730'])
gm_client = gearman.GearmanClient(['localhost:4730'])


def random_filename(file):
    file_name, extension_name = os.path.splitext(file)
    random_name = uuid.uuid4().hex
    # random_name = str(random.randint(1, 10000))
    return file_name + random_name + extension_name


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
        print('收到图片')
        # parse b64 img
        s = gearman_job.data
        img_s = json.loads(s)
        bboxes = img_s['bbox']
        mode = 0

        img_start = time.time()

        d64 = base64.b64decode(img_s['imgs'])
        nparr = np.fromstring(d64, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite('ori.jpg', image)

        img_end = time.time()

        if mode == 0:
            # pose reid
            img_ori_fea_arr = []
            personDB = y.config['personDB']
            persons = []

            start = time.time()
            print(bboxes)

            for i, box in enumerate(bboxes):
                output_dict = face_lib.query_face_db_with_box(image, box)
                # output_dict = face_lib.query_face_db(image)
                # print(output_dict)

                img_name = '{}.jpg'.format(int(time.time()))
                file_name = random_filename(img_name)
                filePath_ori = BASE_DIR + '/' + UPLOAD_FOLDER
                path, date_path = get_state_filepath(filePath_ori, file_name)
                url_path = '/' + UPLOAD_FOLDER + date_path + file_name

                # write img
                # img_copy = image.copy()
                # cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                # cv2.imwrite(path, img_copy)
                cv2.imwrite(path, output_dict['crop_img'])

                print('******')
                print(path)
                print(url_path)
                print('******')
                person = {}
                person['bbox'] = box
                person['img'] = url_path
                person['date'] = get_date_str(time.time())
                if output_dict['id'] != -1:
                    person['rec'] = True
                else:
                    person['rec'] = False
                person['name'] = output_dict['name']
                person['id'] = output_dict['id']
                person['score'] = int(output_dict['score']*100) / 100
                person['timestamp'] = int(img_start*1000)
                persons.append(person)

            # print('score ****')
            # print(face_lib.query_face_db(image)['score'])
            # print('score ****')

            end = time.time()
            print('**********')
            print('img time cost %s' % str(img_end-img_start))
            print('**********')
            print('time cost %s' % str(end-start))
            print('**********')
            #return dic
            if len(persons) == 0:
                person = dict(name="None", img=url_path, rec=False, id=-1, timestamp=int(img_start*1000),
                              score='-1',
                            bbox=bboxes[0], confidence=-1, date=get_date_str(time.time()))
                persons.append(person)
            dic_json = dict(persons=persons, infer_time=(end-start), img_time=(img_end-img_start))
            print(dic_json)

            json_str = json.dumps(dic_json)
            completed_job_request = gm_client.submit_job("DPH_TRACKER_SERVER", json_str, background=True)
            
            postMethod(upload_pic_url, dic_json)

            # with open("./log/log.txt", 'a+', encoding='utf8') as f:
            #     f.write(str(persons))
            #     f.write('\n')

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


def add_persons():
    global y
    personDB = y.config['personDB']

    for i in range(len(personDB['persons'])):
        if 'feature' in personDB['persons'][i].keys():
            continue
        img = cv2.imread(personDB['persons'][i]['img'])

        module1 = pet_engine.MODULES['Face3DKpts']
        keypoint = module1()
        module2 = pet_engine.MODULES['FaceReco_affined']
        face_reco = module2()

        img_vis, pts = keypoint(img, [personDB['persons'][i]['bbox']])
        crop_imgs = get_affine_imgs(pts, img)
        features1 = face_reco(crop_imgs)

        personDB['persons'][i]['feature'] = features1[0][0, :].tolist()
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
        face_lib = FaceLib()
    else:
        channel_id = int(sys.argv[1])
        face_lib = FaceLib()
    print('worker start')
    gm_worker.set_client_id('python-worker')
    gm_worker.register_task('DHP_face', task_listener_reverse)
    gm_worker.work()

    # img_1 = cv2.imread('ori.jpg')
    # output_dict = face_lib.query_face_db(img_1)
    # print(output_dict)
    # output_dict = face_lib.query_face_db_with_box(img_1, [1156, 704, 1238, 788])
    # print(output_dict)
    # add_persons()
    # compare_persons_dbs(0, 1)
