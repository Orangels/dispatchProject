#import cv2
import os, sys
import subprocess
import time
import json

os.environ.update({'DISPLAY': ':1'})


class Fisheye():
    def __init__(self):
        self.pwd = os.path.dirname(os.path.realpath(__file__)) + '/../build'
        self.mapping = {}
        print(self.pwd)

    def build_yaml(self, media_id, status='n', baseMap='n'):
        param = "%YAML:1.0\nstop:" + " %s\npic: %s\n" % (status, baseMap)
        self.yaml = self.pwd + '/param_%d.yaml' % media_id
        with open(self.yaml, 'w')as f:
            f.writelines(param)
        os.system('cat %s' % self.yaml)
        # print(self.yaml)

    def start(self, media_id, media_mac, media_rtsp):
        print("start")
        print(media_id, media_mac, media_rtsp)
        self.mapping[media_id] = {'media_id': media_id, 'media_mac': media_mac, 'media_rtsp': media_rtsp}
        self.build_yaml(media_id)
        os.system("echo $DISPLAY")
        self.runlog = self.pwd + '/run_%d.log' % media_id
        cmd = self.pwd + "/infer_ --ipt %s --opt a --media_id %d --mac %s --yaml %s >%s 2>&1 &" % \
              (media_rtsp, media_id, media_mac, self.yaml, self.runlog)
        print(cmd)
        subprocess.call(cmd, shell=True)
        return True

    def stop(self, media_id, **kwargs):
        print("stop")
        self.runlog = self.pwd + '/run_%d.log' % media_id
        self.build_yaml(media_id, status='y')
        time.sleep(1)
        self._del(media_id)
        return True

    def set(self, media_id, set_type, params, debug=True):
        print("set")
        print(media_id, set_type)
        mapping = self.mapping[media_id]
        sv_media_id = mapping['media_id']
        assert str(media_id) == str(sv_media_id), '%s should equals to old one %s' % (str(media_id), str(sv_media_id))
        mapping['debug'] = debug
        if debug: print("*" * 10)
        params.update(mapping)
        pth = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'set.json')
        with open(pth, 'w')as f:
            f.write(json.dumps(params))

        print('set params:', params, pth)
        return True

    def getpic(self, media_id, **kwargs):
        print("get Base map")
        self.build_yaml(media_id, baseMap='y')
        pic = self.pwd + '/param_%d.jpg' % media_id
        print(pic)
        return True, pic

    def _del(self, media_id):
        print('delete temp files')
        delFile = []  # [self.runlog]  # [self.yaml,]

        for f in delFile:
            if os.path.exists(f):
                print("fils will deleted: rm %s" % f)
                os.system("rm %s" % f)


def use_exists():
    if os.path.exists("/srv/fisheye_prj/AI_Server/utils/py_extension/D_set.json"):
        os.system("cp /srv/fisheye_prj/AI_Server/utils/py_extension/D_set.json "
                  "/home/user/project/run_retina/py_extension/set.json")
    elif os.path.exists('/home/user/project/run_retina/py_extension/D_set.json'):
        os.system("cp /home/user/project/run_retina/py_extension/D_set.json "
                  "/home/user/project/run_retina/py_extension/set.json")


def test_set(debug=False):
    demo = Fisheye()
    media_id = 1
    demo.mapping[media_id] = {'media_id': media_id, 'media_mac': "00-02-D1-83-83-6E",
                              'media_rtsp': 'rtsp://root:admin123@172.16.105.199:554/live2.sdp'}

    params = {'BUSS.COUNT.ROI_AREA_ID': ['11', '7', '5'],
              'BUSS.COUNT.ROI_DOTEED_LINE_AREA': {
                  '7': [[717, 169], [758, 260], [980, 238], [979, 156]],
                  '11': [[640, 235], [626, 266], [1024, 678], [1022, 247]],
                  '5': [[948, 496], [965, 616], [1105, 565], [1078, 513]]},
              'BUSS.COUNT.ROI_SOLID_LINE_AREA': {
                  '7': [[760, 219], [737, 345], [1027, 335], [999, 264]],
                  '11': [[663, 123], [633, 224], [1013, 234], [1030, 123]],
                  '5': [[905, 488], [962, 648], [1069, 675], [1132, 581], [1123, 520], [1024, 457]]},
              'BUSS.COUNT.ROI_AREA_TYPE': {'5': 3, '11': 1, '7': 1}}
    demo.set(media_id, 'ALG', params, debug=debug)


if __name__ == '__main__':
    test_set(True)  # True
    demo = Fisheye()
    status = sys.argv[1]
    md = int(sys.argv[2])
    local_video = '/../photos/sjj.mp4'  # ''  #bug.avi

    if status == 's':  # start
        demo.start(md, "00-02-D1-83-83-6E", 'rtsp://root:admin123@172.16.105.199:554/live1.sdp')
    elif status == 'd':  # start
        demo.start(md, "'00-02-D1-83-83-71'", demo.pwd + local_video)
    elif status == 'e':  # end
        demo.stop(md)
    else:
        demo.getpic(md)
    # # ╰─$ build/infer_ --ipt ./test_vedio/s.mp4 --opt 'a' --media_id 1 --mac 'werqwr:345' --yaml build/param_1.yaml
