from flask import Flask, jsonify, request, send_file, send_from_directory, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS

import time
from os import popen
from urllib.parse import unquote
import sys
import cv2

import os
import json
from werkzeug.routing import BaseConverter
import random
import traceback
from tools.utils import *


class RegexConverter(BaseConverter):
    def __init__(self, map, *args):
        self.map = map
        self.regex = args[0]


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # 保存文件位置
app.url_map.converters['regex'] = RegexConverter

socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources=r'/*')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
@app.route('/mode_1')
@app.route('/mode_2')
@app.route('/login')
def index():
    # return 111
    return render_template("index.html")


@app.route('/waring_img', methods=['POST'])
def post_new_waring_img():
    try:
        data = unquote(request.data.decode(), 'utf-8')
        client_params = json.loads(data)['persons']
        # print(client_params)
        socketio.emit('new_state', {
            'result': client_params
        },
                      namespace='/Camera_Web_ws')
        dic = dict(code=200, result='error')
        return jsonify(dic)
    except Exception as e:
        print('***********')
        print(e)
        traceback.print_exc()
        print('***********')
        dic = dict(code=400, result='error')
        return jsonify(dic)


@app.route('/device_info', methods=['POST'])
def device_info():
    try:
        dic = dict(code=200, result=get_device_basic_info())
        return jsonify(dic)
    except Exception as e:
        print('***********')
        print(e)
        traceback.print_exc()
        print('***********')
        dic = dict(code=400, result='error')
        return jsonify(dic)


@socketio.on('test_message', namespace='/Camera_Web_ws')
def test_message(message):
    print(message)
    emit('my response', {'data': message['data']})


@socketio.on('my broadcast event', namespace='/Camera_Web_ws')
def test_message(message):
    print('my broadcast event')
    emit('my response', {'data': message['data']}, broadcast=True)


@socketio.on('connect', namespace='/Camera_Web_ws')
def test_connect():
    print('Web connected')


@socketio.on('disconnect', namespace='/Camera_Web_ws')
def test_disconnect():
    print('Web disconnected')


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000, debug=True)
    socketio.run(app, host='0.0.0.0', port=5000)
