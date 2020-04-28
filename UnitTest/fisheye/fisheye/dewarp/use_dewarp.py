import ctypes
import os, cv2, time, math
import numpy as np
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='dewarp fisheye')
parser.add_argument('--f', dest='files', default='', help='file names using separate by ","', type=str)
parser.add_argument('--r', dest='roots', default='', help='root of those file(optinal)', type=str)
parser.add_argument('--s', dest='stops', default=None, help='how many frames to deal with (optinal)', type=int)

kernel = './dewarper.so'
print('kernel:', kernel)
lib = ctypes.cdll.LoadLibrary(kernel)


def get_size(FLAGS_mode=b"center", FLAGS_pixels_per_degree=16.0,
             FLAGS_center_zoom_angle=90.0,
             FLAGS_perimeter_top_angle=90,
             FLAGS_perimeter_bottom_angle=30.0):
    print("mode---", FLAGS_mode)
    if (FLAGS_mode == b"center"):
        length = int(FLAGS_pixels_per_degree * FLAGS_center_zoom_angle / 4) << 2
        print("py size: ", length)
        return length, length
    else:
        width = int(FLAGS_pixels_per_degree * 180.0 / 4) << 2
        height = int(FLAGS_pixels_per_degree *
                     (FLAGS_perimeter_top_angle - FLAGS_perimeter_bottom_angle)
                     / 2) << 2
        # print ("py size: W: ", width, " h: ", height)
        return width, height


def model(rtmp, pixels=16, stop_at=None):
    allmode = [b"perimeter", b"center"]
    lens = b"vivotek"  # b"3s"
    rota = 0  # -49
    top_angle = 95
    bottom_angel = 30
    zoom_angle = 90
    for mode in allmode:
        image_id = 0
        w_hat, h_hat = get_size(FLAGS_mode=mode, FLAGS_pixels_per_degree=pixels)
        print("size w&h", w_hat, h_hat)
        name = str(mode, encoding="utf-8")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vedio_code = rtmp.split('/')[-1].split('.')
        vedio_name = [*vedio_code[:-1], name, vedio_code[-1]]
        vedio_code = '.'.join(vedio_name)
        print('save as:results/%s' % vedio_code, end=', ')
        vedio = cv2.VideoWriter('results/' + vedio_code + '.avi',
                                fourcc, 30.0, (w_hat, h_hat))
        cap = cv2.VideoCapture(rtmp)
        total = cap.get(7)  # CV_CAP_PROP_FRAME_COUNT 7
        print('and has %d frames' % total)
        out = np.zeros((h_hat, w_hat, 3), dtype=np.uint8)
        if stop_at is None:
            stop_at = total
        else:
            stop_at = max(stop_at, 10)
        while True:
            ret, img = cap.read()
            if image_id % 20 == 0:
                print('%d/%d' % (image_id, total), end=', ')
            if ret:
                if image_id > stop_at:
                    break
                h, w, _ = img.shape
                image_id += 1
                dataptr_in = img.ctypes.data_as(ctypes.c_char_p)
                outptr_in = out.ctypes.data_as(ctypes.c_char_p)
                status = lib.run(dataptr_in, h, w,
                                 outptr_in, h_hat, w_hat,
                                 image_id, lens, mode,
                                 rota, zoom_angle, top_angle, bottom_angel)

                vedio.write(out)
            else:
                break
        print('\nDone!')
        cap.release()
        vedio.release()
        print('- ' * 10)


if __name__ == "__main__":
    args = parser.parse_args()
    # print(args)
    root = args.roots.strip()
    names = [os.path.join(root, x.strip()) for x in args.files.split(',')]
    print(names)
    pixeslPdegree = 16
    for name in names:
        if not ('mp4' in name or 'avi' in name):
            continue
        print('##' * 10, name, '##' * 10)
        model(name, pixels=pixeslPdegree, stop_at=args.stops)
    # print(get_size())
