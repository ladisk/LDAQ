from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from simple_websocket_server import WebSocket
import webbrowser
import json

import numpy as np

class Visualization:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'secret!'
        self.app.config['UPGRADE_WEBSOCKET'] = WebSocket
        self.socketio = SocketIO(self.app, async_mode='threading')

        self.running = False
        
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.socketio.on('connect')
        def on_connect():
            print('Client connected')

        @self.socketio.on('disconnect')
        def on_disconnect():
            print('Client disconnected')

        @self.socketio.on('start')
        def on_start():
            self.running = True
            self.socketio.start_background_task(self.generate_data)

        @self.socketio.on('stop')
        def on_stop():
            print('stop')
            self.running = False

        @self.socketio.on('close')
        def on_disconnect():
            print('Client closed')


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
        webbrowser.open('http://localhost:5000')
        self.socketio.run(self.app)


if __name__ == '__main__':
    visualizer = Visualization()
    visualizer.run()
