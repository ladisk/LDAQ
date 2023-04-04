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

logger = logging.getLogger()
logging.basicConfig(filename='visualization.log', level=logging.INFO)


class Visualization:
    def __init__(self):
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
            logger.info('Client connected')

        @self.socketio.on('disconnect')
        def on_disconnect():
            logger.info('Client disconnected')

        @self.socketio.on('start')
        def on_start():
            self.running = True
            self.socketio.start_background_task(self.generate_data)

        @self.socketio.on('stop')
        def on_stop():
            logger.info('stop')
            self.running = False

        @self.socketio.on('close')
        def on_close():
            from win32api import GenerateConsoleCtrlEvent
            CTRL_C_EVENT = 0
            GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0)
            logger.info('Client closed')


    def generate_data(self):
        x = np.arange(1000)
        y = np.zeros(1000)
        while True:
            if self.running:
                y_new = np.random.rand(100)
                y = np.roll(y, -len(y_new))
                y[-len(y_new):] = y_new
                data = {
                    'x': x.tolist(),
                    'y': y.tolist()
                }
                
                self.socketio.emit('data', json.dumps(data))
                self.socketio.sleep(0.1)
            else:
                self.running = False
                break


    def run(self):
        webbrowser.open('http://127.0.0.1:5000')
        self.socketio.run(self.app, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    visualizer = Visualization()
    visualizer.run()
