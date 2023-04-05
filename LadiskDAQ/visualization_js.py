from flask import Flask, render_template
from flask import request
from flask_socketio import SocketIO, emit
from simple_websocket_server import WebSocket
from werkzeug.serving import make_server
import webbrowser
import json
from threading import Thread
import time
import subprocess
import sys
import os
import numpy as np
import logging
import msgpack  # Install the package using: pip install msgpack

# logger = logging.getLogger()
# logging.basicConfig(filename='visualization.log', level=logging.INFO)
# print_log = logger.info
print_log = print


class Visualization:
    def __init__(self, layout=None, subplot_options=None):
        self.layout = layout
        self.subplot_options = subplot_options

        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'sadfahsdgijweoifjačlrgijačlkgjčlkj234512352315'
        self.app.config['UPGRADE_WEBSOCKET'] = WebSocket
        # self.socketio = SocketIO(self.app, async_mode='threading')
        self.socketio = SocketIO(self.app)

        self.running = False
        
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.socketio.on('connect')
        def on_connect():
            print_log('Client connected')
            # self.socketio.emit('layout', json.dumps(self.layout))

        @self.socketio.on('disconnect')
        def on_disconnect():
            print_log('Client disconnected')

        @self.socketio.on('start')
        def on_start():
            self.running = True
            self.socketio.start_background_task(self.update_data)

        @self.socketio.on('stop')
        def on_stop():
            print_log('stop')
            self.running = False

        @self.socketio.on('close')
        def on_close():
            from win32api import GenerateConsoleCtrlEvent
            CTRL_C_EVENT = 0
            GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0)
            print_log('Client closed')


    def update_data(self):
        x = np.arange(1000)
        y = np.zeros(1000)
        data = {'NI_task': {}}
        while True:
            if self.running:
                # data = self.core.get_measurement_dict(5.)
                # for k in data:
                #     data[k]['data'] = data[k]['data'].T.tolist()
                #     data[k]['time'] = data[k]['time'].tolist()
                
                data['NI_task']['data'] = np.random.rand(10000, 10).T.tolist()
                data['NI_task']['time'] = np.arange(10000).tolist()
                binary_data = msgpack.packb(data, use_bin_type=True)
                self.socketio.emit('data', binary_data)
                self.socketio.sleep(0.1)
            else:
                self.running = False
                break


    def run(self, core):
        self.core = core
        webbrowser.open('http://127.0.0.1:5000')
        self.socketio.run(self.app, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    visualizer = Visualization()
    visualizer.run()
