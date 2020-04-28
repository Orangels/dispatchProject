import cv2
import sys
import pymongo
import numpy as np
import os.path as osp

import torch

import _init_paths
from core.config import get_cfg_defaults

class FaceLib:
    def __init__(self):
        this_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
        print(osp.join(this_dir, 'cfgs/face_lib.yaml'))
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file(osp.join(this_dir, 'cfgs/face_lib.yaml'))
        self.cfg.freeze()
        print(self.cfg)

        self.init_db()
        if self.cfg.PET_ENGINE.USE_ENGINE:
            self.init_pet_engine()

        self.cos_dis = torch.nn.CosineSimilarity()

    def init_db(self):
        face_lib_client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
        face_lib_db = face_lib_client[self.cfg.FACE_LIB.DB_NAME]
        self.face_lib_col = face_lib_db[self.cfg.FACE_LIB.COL_NAME]
        sort_all = self.face_lib_col.find().sort('id')
        self.ids = []
        features = []
        for i in sort_all:
            self.ids.append(i['id'])
            features.append(i['feature'])
        self.features = torch.from_numpy(np.array(features)).cuda()
        self.max_id = self.ids[-1]

    def init_pet_engine(self):
        sys.path.append(self.cfg.PET_ENGINE.PATH)
        from modules import pet_engine

        module = pet_engine.MODULES['ObjectDet']
        self.det = module(cfg_file=self.cfg.PET_ENGINE.CFG)
        module = pet_engine.MODULES['Face3DKpts']
        self.face_3dkpts = module()
        module = pet_engine.MODULES['FaceReco']
        self.face_reco = module()

    def get_crop_img(self, img, box):
        x = box[0]
        y = box[1]
        w = box[2] - x
        h = box[3] - y
        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        src_w = max(w, h) * self.cfg.CROP_IMG.SCALE
        dst_w = dst_h = self.cfg.CROP_IMG.SIZE

        src_dir = np.array([0, src_w * -0.5], np.float32)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        crop_img = cv2.warpAffine(
            img,
            trans,
            (dst_w, dst_h),
            flags=cv2.INTER_LINEAR
        )
        return crop_img

    def query_face_db(self, img):
        output_det = self.det(img)
        im_labels = np.array(output_det['im_labels'])
        im_dets = output_det['im_dets'][im_labels == self.cfg.PET_ENGINE.FACE_DET_ID]
        bbox = im_dets[:, :4]
        score = im_dets[:, -1]
        sorted_inds = np.argsort(-score)
        face_box = bbox[sorted_inds[:1]]
        face_score = score[sorted_inds[:1]]

        if face_score < self.cfg.PET_ENGINE.FACE_DET_TH:
            return {'corp_img': None}

        return self.query_face_db_with_box(img, face_box[0])

    def query_face_db_with_box(self, img, face_box):
        crop_img = self.get_crop_img(img, face_box)
        output_reco = self.face_reco(img, [face_box])['features']
        dis = self.cos_dis(output_reco, self.features)
        s, ind = dis.max(dim=0)
        output_dict = {'crop_img': crop_img, 'score': float(s.cpu())}
        if s > self.cfg.FACE_LIB.MATCH_TH:
            db_dict = self.face_lib_col.find({'id': self.ids[ind]}, {'feature': 0})[0]
            output_dict.update(db_dict)
        else:
            output_dict.update({'id': -1, 'name': 'None'})
        return output_dict


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


if __name__ == '__main__':
    face_lib = FaceLib()
    img = cv2.imread('/home/user/Program/Share/wzh/spider_face/images/陆毅/000001.jpg')
    # img = cv2.imread('/home/user/Program/Share/wzh/spider_face/images/姜文/000001.jpg')
    output_dict = face_lib.query_face_db(img)
    # cv2.imwrite('/home/user/Program/Share/wzh/Pet-face-lib/data/luyi.jpg', output_dict['corp_img'])

    img_1 = cv2.imread('/home/user/Program/Share/wzh/privision_test/1661.jpg')
    output_dict = face_lib.query_face_db_with_box(img_1, [468, 220, 634, 413])
    # cv2.imwrite('/home/user/Program/Share/wzh/Pet-face-lib/data/wzh.jpg', output_dict['corp_img'])

