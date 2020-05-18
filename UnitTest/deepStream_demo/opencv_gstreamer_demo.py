# coding=utf-8

import sys
import argparse
import cv2
import numpy as np
import time


def open_cam_rtsp(uri, width, height, latency,use_h264):
    uri_str=uri[0:4]
    if (uri_str != "rtsp"):
        if (uri_str != "rtmp"):
            is_file = 1
        return cv2.VideoCapture(uri)

    if use_h264:
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! omxh264dec ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(uri, latency, width, height)
    else:
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph265depay ! h265parse ! omxh265dec ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_cam_usb(dev, width, height):
    gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)RGB ! '
               'videoconvert ! appsink').format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_cam_onboard(width, height):
    gst_str = ('nvcamerasrc ! '
               'video/x-raw(memory:NVMM), '
               'width=(int)2592, height=(int)1458, '
               'format=(string)I420, framerate=(fraction)30/1 ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


if __name__ == "__main__":

    image_width = 640
    image_height = 480
    rtsp_latency = 0

    uri = "rtsp://admin:sx123456@192.168.88.37:554/h264/ch2/sub/av_stream"
    # gst_str = (
    #     "rtspsrc location={} latency={} ! rtph264depay ! h264parse ! avdec_h264 ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! appsink sync=false").format(
    #     uri, rtsp_latency, image_width, image_height)

    # gst_str = 'rtspsrc location=rtsp://admin:sx123456@192.168.88.37:554/h264/ch2/sub/av_stream latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink sync=false'
    # gst_str = 'rtspsrc location=rtsp://admin:sx123456@192.168.88.37:554/h264/ch2/sub/av_stream latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, width=(int)640, height=(int)360, format=(string)BGRx ! videoconvert ! appsink'
    # gst_str = 'rtspsrc location=rtsp://root:admin123@192.168.88.26/live.sdp latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, width=(int)640, height=(int)360, format=(string)BGRx ! videoconvert ! appsink'
    # gst_str = 'rtspsrc location=rtsp://root:admin123@192.168.88.67/live.sdp latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, width=(int)640, height=(int)360, format=(string)BGRx ! videoconvert ! appsink'
    # gst_str = 'rtsp://root:admin123@192.168.88.26/live.sdp'
    gst_str = 'rtsp://admin:sx123456@192.168.88.37:554/h264/ch2/sub/av_stream'

    cap = cv2.VideoCapture(gst_str)
    # cap = cv2.VideoCapture(uri)
    # if not cap.isOpened():
    #     sys.exit("Failed to open camera!")
    # else:
    #     suc, frame = cap.read()
    #     cv2.imwrite('test.jpg', frame)
    suc, frame = cap.read()
    frame_count = 0
    time_sum = 0
    while suc:
        frame_count += 1
        time_start = time.time()
        suc, frame = cap.read()
        print('{} cost {}'.format(frame_count, time.time()-time_start))
        time_sum += time.time()-time_start
        if frame_count == 100:
            print(time_sum/frame_count)
            cv2.imwrite('test.jpg', frame)
            break







