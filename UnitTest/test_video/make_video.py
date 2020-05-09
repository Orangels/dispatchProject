import cv2
import sys
import time

fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')  # 保存 mp4
path = ''
out_path = 'asd.mp4'
video_time = 1*60*20


def make_video():
    global path
    global out_path
    cap = cv2.VideoCapture(path)
    suc = cap.isOpened()  # 是否成功打开
    # 帧数
    frame_count = 0
    # 轨迹容错帧数
    jump_frame_count = 0

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    suc, frame = cap.read()
    frame_count = 0

    videoWriter = cv2.VideoWriter(out_path, fourcc, 20, (frame.shape[1], frame.shape[0]))
    videoWriter.write(frame)
    while suc:
        start = time.time()
        frame_count += 1
        suc, frame = cap.read()
        videoWriter.write(frame)
        print('{} : {} time cost -- {}'.format(out_path, frame_count, time.time()-start))
        if frame_count > video_time:
            break
    videoWriter.release()


if __name__ == "__main__":
    mode = int(sys.argv[1])
    if mode == 0:
        path = 'rtsp://192.168.88.29:554/user=admin&password=admin&channel=1&stream=0.sdp?real_stream'
        out_path = 'gate_camera.mp4'
    elif mode == 1:
        path = 'rtsp://root:admin123@192.168.88.67/live.sdp'
        out_path = 'fish_0.mp4'
    else:
        path = 'rtsp://root:admin123@192.168.88.26/live.sdp'
        out_path = 'fish_1.mp4'
    make_video()
