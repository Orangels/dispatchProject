# encoding: utf-8
import base64
import json
import paho.mqtt.client as mqtt
import numpy as np


# ffmpeg 推流：https://blog.csdn.net/Mind_programmonkey/article/details/102732555
class connectDB():
    def __init__(self, debug=False):
        self.client = mqtt.Client()
        try:
            self.client.connect("127.0.0.1", 1883, 60)
        except Exception as e:
            pass
        self.idx = 0
        self.debug = debug

    def push_out(self, media_id, media_mac, image_id, image):
        if self.debug and self.idx < 5:
            self.idx += 1
            np.savez('DB_%d' % self.idx, i=image)
        if isinstance(image, np.ndarray):
            # to base64: https://blog.csdn.net/wangjian1204/article/details/84445334
            b64_code = base64.b64encode(image.tostring()).decode()  # 编码成base64
            value = {'media_id': media_id,
                     'media_mac': media_mac,
                     'picfile': b64_code,
                     'image_id': image_id,
                     'format': "image/jpeg"}
            param = json.dumps(value)
            # https://www.eclipse.org/paho/files/mqttdoc/MQTTClient/html/struct_m_q_t_t_client__message.html#a35738099155a0e4f54050da474bab2e7
            try:
                self.client.publish(media_mac, param, 0)
            except Exception as e:
                pass

