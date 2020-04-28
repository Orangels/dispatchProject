import sys
import cv2
import numpy as np
from collections import defaultdict

import torch

sys.path.append('/home/user/Program/Share/wzh/Pet-engine')
from modules import pet_engine
from projects.face_3dkeypoints.utils.estimate_pose import matrix2angle
from utils.timer import Timer


def compute_dist(array1, array2):
    array3 = np.sqrt(np.power(array1, 2).sum(axis=1, keepdims=True))
    array4 = np.sqrt(np.power(array2, 2).sum(axis=1, keepdims=True))
    if np.dot(array3, np.transpose(array4)).any() == 0:
        return -1
    similarity = np.dot(array1, np.transpose(array2)) / np.dot(array3, np.transpose(array4))

    return similarity


if __name__ == '__main__':
    print(pet_engine.MODULES.keys())
    # timers = defaultdict(Timer)
    # module = pet_engine.MODULES['ObjectDet']
    # det = module(cfg_file='hf.yaml')
    # img = cv2.imread('P000566.png')
    # for i in range(100):
    #     img_vis = det(img)
    # # cv2.imwrite('222.png', img_vis)

    # module = pet_engine.MODULES['Face3DKpts']
    # face = module()
    # img = cv2.imread('000000255800.jpg')
    # output_kpts = face(img, [[187, 59, 243, 131]])
    # print(output_kpts)
    # x, y, z = matrix2angle(output_kpts['p'][0])
    # print(x, y, z)
    #
    # module = pet_engine.MODULES['FaceReco']
    # face = module()
    # img_1 = cv2.imread('1661.jpg')
    # img_2 = cv2.imread('3012.jpg')
    # timers['all_reco'].tic()
    # for i in range(500):
    #     timers['reco'].tic()
    #     output1 = face(img_1, [[468, 220, 634, 413]])['features']
    #     output2 = face(img_2, [[535, 290, 690, 455]])['features']
    #     timers['reco'].toc()
    # timers['all_reco'].toc()
    # print(' | {}: {:.3f}s'.format('reco', timers['reco'].average_time / 2.0))
    # print(' | {}: {:.3f}s'.format('all_reco', timers['all_reco'].average_time / 1000.0))
    # module = pet_engine.MODULES['FaceReco']
    # face = module()
    # img_1 = cv2.imread('1661.jpg')
    # img_2 = cv2.imread('3012.jpg')
    # output1 = face(img_1, [[468, 220, 634, 413]])['features']
    # # output1 = output1.expand(7999, -1)
    # output1 = output1.repeat(1999, 1)
    # output2 = face(img_2, [[535, 290, 690, 455]])['features']
    # output1 = torch.cat([output1, output2])
    #
    # output1_n = output1.cpu().numpy()
    # output2_n = output2.cpu().numpy()
    #
    # cos = torch.nn.CosineSimilarity()
    #
    # timers['all_dis_n'].tic()
    # for i in range(1000):
    #     timers['numpy'].tic()
    #     dis_1 = compute_dist(output2_n, output1_n)
    #     timers['numpy'].toc()
    # timers['all_dis_n'].toc()
    # print(' | {}: {:.3f}s'.format('numpy', timers['numpy'].average_time))
    # print(' | {}: {:.3f}s'.format('all_dis_n', timers['all_dis_n'].average_time / 1000.0))
    #
    # timers['all_dis_c'].tic()
    # for i in range(1000):
    #     timers['cuda'].tic()
    #     dis_1 = cos(output2, output1)
    #     timers['cuda'].toc()
    # timers['all_dis_c'].toc()
    # print(' | {}: {:.3f}s'.format('cuda', timers['cuda'].average_time))
    # print(' | {}: {:.3f}s'.format('all_dis_c', timers['all_dis_c'].average_time / 1000.0))

    module = pet_engine.MODULES['FaceReco']
    face = module()
    img = cv2.imread('/home/user/Program/ls-dev/dispatchProject/UnitTest/engine/EngineServer/ori.jpg')
    A = face(img, [[1156, 704, 1238, 788]])['features'].cpu().numpy().astype(np.float32)
    img = cv2.imread('/home/user/Program/ls-dev/dispatchProject/UnitTest/engine/EngineServer/imgs/ls.jpg')
    B = face(img, [[274, 212 ,435, 435]])['features'].cpu().numpy().astype(np.float32)
    img = cv2.imread('/home/user/Program/Share/wzh/privision_test/C.jpg')
    C = face(img, [[0, 0, 315, 311]])['features'].cpu().numpy().astype(np.float32)

    print('Jack  vs Jack ', np.dot(A[0, :], B[0, :]))
    print('Jack  vs Jun ', np.dot(A[0, :], C[0, :]))


    # module = pet_engine.MODULES['SSDDet']
    # hop_det = module(cfg_file='/home/wangzhihui/Pet-engine/cfgs/hand_det.yaml')
    #
    # img = cv2.imread('frame_0443.jpg')
    # result = hop_det(img)
    # print(result)