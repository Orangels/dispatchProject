import numpy as np
# import cv2
# from threading import Thread
# from multiprocessing import Process
# import ctypes

# from apis.face_lib import get_max_iou
# __all__ = ['rearrange_box', 'get_equal_box_id', 'split_for_no', 'npbbox_iou',
#            'split_img', 'split_img_twoparts', 'no_split', 'hello']  # 'run_action',


# def action(img, FaceDetKps):
#     img_resize = np.zeros([4], np.uint8)
#     cls_boxes_i = FaceDetKps.ssd_det(img, img_resize)

#
# def trac_thread_action(FaceDetKps):
#     '''
#     aim to pack the tracking function of facelib here
#     '''
#     out = {}
#
#     # use_boxes_filter = True
#     # ssd = FaceDetKps.ssd_det.model
#     def _action(img, img_id):
#         outputs_ = FaceDetKps.ssd_det(img, 0)
#         out[img_id] = outputs_
#
#     # def _action3(img, img_id):
#     #     max_detection_num = 400
#     #     NUM_CLASS = 3
#     #     outputs_ = []
#     #     outputs = np.zeros([max_detection_num, 6], np.float32)
#     #
#     #     numDetection = eval('ssd.model.' + ssd.model_name)(img.shape[0], img.shape[1],
#     #                                                        img.ctypes.data_as(ctypes.c_void_p),
#     #                                                        outputs.ctypes.data_as(ctypes.c_void_p),0)
#     #                                                        # img_resize.ctypes.data_as(ctypes.c_void_p))
#     #     outputs = outputs[:numDetection]
#     #     # print(outputs)
#     #     for class_id in range(NUM_CLASS):
#     #         current_id = (outputs.T[5]).astype(np.int64) == class_id
#     #         outputs_.append(outputs[current_id, 0:5])
#     #     out[img_id] = outputs_
#
#     return _action, out
#
#
# def run_action(FaceDetKps, imgs):
#     tgt, out = trac_thread_action(FaceDetKps)
#
#     threads = []
#     for i, img in enumerate(imgs):
#         tgt(img, i)
#     for i, img in enumerate(imgs):
#         t1 = Thread(target=tgt, args=(img, i))
#         threads.append(t1)
#     for t in threads:
#         t.setDaemon(True)
#         t.start()
#     for t in threads:
#         t.join()
#     return out


# def async_call(fn):
#     def wrapper(*args, **kwargs):
#         # print('dddd')
#         Thread(target=fn, args=args, kwargs=kwargs).start()
#
#     return wrapper


# @async_call
# def hello(img, fun, image_id, out_dict):
#     cls_boxes_i = fun(img, 0)
#     out_dict[image_id] = cls_boxes_i
#     # return cls_boxes_i


def npbbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    n, items1 = box1.shape
    m, items2 = box2.shape
    # box1_clone = boxes1.reshape(n, 1, items1)

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0:1], box1[:, 1:2], box1[:, 2:3], box1[:, 3:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    # np.tile(b1_x1, (1, m)) np.tile(b2_x1, (n,1))
    # model = np.ones((1,n,  m,items2), dtype=box1.dtype)
    model = np.ones((n, m), dtype=box1.dtype)
    bd_model = np.zeros((n, m), dtype=box1.dtype)
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = np.max(np.stack([b1_x1 * model, b2_x1 * model]), 0)  # np.max(b1_x1, b2_x1)
    inter_rect_y1 = np.max(np.stack([b1_y1 * model, b2_y1 * model]), 0)  # np.max(b1_y1, b2_y1)
    inter_rect_x2 = np.min(np.stack([b1_x2 * model, b2_x2 * model]), 0)  # np.min(b1_x2, b2_x2)
    inter_rect_y2 = np.min(np.stack([b1_y2 * model, b2_y2 * model]), 0)  # np.min(b1_y2, b2_y2)

    inter_rect_x = inter_rect_x2 - inter_rect_x1 + 1
    inter_rect_y = inter_rect_y2 - inter_rect_y1 + 1
    # inter_rect_x = inter_rect_x.reshape(model.shape)
    # inter_rect_y = inter_rect_y.reshape(model.shape)
    # Intersection area

    inter_area = np.max(np.stack([inter_rect_x, bd_model]), 0) * \
                 np.max(np.stack([inter_rect_y, bd_model]), 0)
    # inter_area = inter_area.reshape(model.shape)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

'''
def rearrange_box(detect_box_N_imgs, span=1704, boundary_area=200, merge_as_one=True):
     
    #aim to arrange boxes of sub_images being a bigger series
     

    def add_offset(cls_boxes, offset):
         
        #let all rectangle move right by "offset"
        
        for boxes in cls_boxes:
            try:
                boxes[:, :3:2] += offset
            except Exception as e:
                pass

    def merge_same_instance(cls_boxes1, cls_boxes2):
        # todo:but how to deal with same instance locate between upper and lower
        # let box locate same area merging into one.
        # those meet the demand should almost have same boxes.
        # in fact, no solution for those having different box but belonging to same instance
        cls_boxes, left_items = [], []
        for box_1, box_2 in zip(cls_boxes1, cls_boxes2):
            boxes1, boxes2 = box_1.copy(), box_2.copy()
            if len(boxes2) and len(boxes1):
                print()
            esp = 10  # total diff of four coner less than 10 pixels
            box1_out, box2_out = get_equal_box_id(boxes1, boxes2, esp)
            if box1_out is not None:
                boxes1 = boxes1[box1_out]
                boxes2 = boxes2[box2_out]
            stack = np.vstack([boxes1, boxes2]) 
            cls_boxes.append(stack)
            left_items.append(boxes1.shape[0])
        return cls_boxes, left_items

    # upper, lower = [0, 1], [2, 3]
    half_ = span - boundary_area
    for right in [1, 3]:
        offset = half_
        add_offset(detect_box_N_imgs[right], offset)
    upper_img, _ = merge_same_instance(detect_box_N_imgs[0], detect_box_N_imgs[1])
    lower_img, _ = merge_same_instance(detect_box_N_imgs[2], detect_box_N_imgs[3])

    total_ = half_ * 2

    return upper_img, lower_img, total_ + boundary_area


def get_equal_box_id(box_1, box_2, esp=2, mapping=False):
    # out1 :those > esp is True; out2: idx of box2 when out1== False
    # todo:here!!
    thres = 0.45
    boxes1, boxes2 = box_1.copy(), box_2.copy()
    if not (len(boxes1) or len(boxes2)):
        return None, None
    n, items1 = boxes1.shape
    m, items2 = boxes2.shape
    if n > 1 and m > 1:
        print()
    if n * items1 * m * items2:
        nms = npbbox_iou(boxes1, boxes2)
        idx = np.argmax(nms, -1)
        value = nms[range(len(nms)), idx]
        if mapping:
            map_ = value > thres
            map_2 = idx[map_]
            map_1 = np.arange(len(map_))[map_]
            return map_1, map_2
        ######################
        # out_1 = value < thres
        # out_2 = idx[~out_1]
        ######################
        out1, out2 = [], []
        # get high score id
        for i, v in enumerate(value):
            if v > thres:
                # conflict
                score1 = boxes1[i, 4]
                score2 = boxes2[idx[i], 4]
                if score1 > score2:
                    out1.append(i)
                    out2.append(idx[i])  # to remove

            else:
                # not conflict
                out1.append(i)
        out2 = list(set(list(range(len(boxes2)))) - set(out2))
        out_1 = np.array(out1, dtype=np.uint8)
        out_2 = np.array(out2, dtype=np.uint8)
        return out_1, out_2
    else:
        return None, None


def passss(boxes1, boxes2):
    thres = 0.5
    nms = npbbox_iou(boxes1, boxes2)
    idx = np.argmax(nms, -1)
    value = nms[range(len(nms)), idx]
    # out_1 = value > thres
    # out_2 = idx[out_1]
    out1, out2 = [], []
    # get high score id
    for i, v in enumerate(value):
        if v > thres:
            score1 = boxes1[i, 4]
            score2 = boxes2[idx[i], 4]
            if score1 > score2:
                out1.append(i)
            else:
                out2.append(idx[i])
    out1 = set(list(range(len(value)))) - set(out1)
    out2 = np.array(out2)

    # return out_1, out_2


def split_for_no(dete_out, H):
    up, down = [], []
    midH = H // 2
    for box in dete_out:
        up_box = (box[:, 3] - midH) < 0.1 * (box[:, 3] - box[:, 1])
        up.append(box[up_box])
        down_box = (midH - box[:, 1]) < 0.1 * (box[:, 3] - box[:, 1])
        down.append(box[down_box])
    return up, down


def split_img(img, off=200):
     
    #fisheye img using "3s" will split to 4 sub-imgs for now
     
    out = []
    two = []
    H, W, c = img.shape
    mid_W = W // 2
    # todo:alert! here
    # img = img[::-1].copy()

    # only right
    # out.append(img[:H // 2, :mid_W + off].copy())
    # out.append(img[:H // 2, mid_W - off:W].copy())
    # out.append(img[H // 2:H, :mid_W + off].copy())
    # out.append(img[H // 2:H, mid_W - off:W].copy())

    # both side
    off = off // 2
    # for every one
    # out.append(np.hstack([img[H // 2:, W - off:], img[:H // 2, :mid_W + off]]))  # left
    # out.append(np.hstack([img[:H // 2, mid_W - off:], img[H // 2:, :off]]))  # right
    # out.append(np.hstack([img[:H // 2, W - off:], img[H // 2:, :mid_W + off]]))  # left
    # out.append(np.hstack([img[H // 2:, mid_W - off:], img[:H // 2, :off]]))  # right

    # make bigger first, then every one
    tmp = np.hstack((img[H // 2:, W - off:], img[:H // 2], img[H // 2:], img[:H // 2, :off]))
    # print(tmp.shape)
    out = [tmp[:, :mid_W + 2 * off].copy(),
           tmp[:, mid_W:W + 2 * off].copy(),
           tmp[:, W:  W + mid_W + 2 * off].copy(),
           tmp[:, W + mid_W:].copy()]
    # save to view
    # for i, pht in enumerate(out):
    #     cv2.imwrite('/home/lidachong/VisionProject/results/x%d.jpg' % i, pht)
    return out


#def split_img_twoparts(img, off=200, fx=True):
#     
#    #fisheye img using "3s" will split to 4 sub-imgs for now
#     
#    out = []
#    H, W, c = img.shape
#    if off < 1: off = (int(off * W)) // 2 * 2
#    # todo:alert! here
#    # img = img[::-1].copy()
#
#    off = off // 2
#    up = np.hstack([img[H // 2:, W - off:], img[:H // 2], img[H // 2:, :off]])
#    dn = np.hstack([img[:H // 2, W - off:], img[H // 2:], img[:H // 2, :off]])
#    h, w, c = up.shape
#    if fx == True:
#        out.append(cv2.resize(up, (w // 2, h)))  # up
#        out.append(cv2.resize(dn, (w // 2, h)))  # down
#    else:
#        out = [up, dn]
#    return out


def no_split(img, off=200):
    H, W, c = img.shape
    off = off // 2
    out = np.hstack([img[:, W - off:], img, img[:, :off]])
    return [out]
'''
