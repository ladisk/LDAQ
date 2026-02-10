import random
import secrets
import os
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from simple_websocket_server import WebSocket
import numpy as np
import webbrowser
import json

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
app.config['UPGRADE_WEBSOCKET'] = WebSocket
socketio = SocketIO(app, async_mode='threading')

RUNNING = False

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def on_connect():
    print('Client connected')

@socketio.on('disconnect')
def on_disconnect():
    print('Client disconnected')

@socketio.on('close')
def on_disconnect():
    print('Client closed')

def generate_data():
    global RUNNING
    x = np.arange(1000)
    y = np.zeros(1000)
    while True:
        if RUNNING:
            y_new = np.random.rand(100)
            y = np.roll(y, -len(y_new))
            y[-len(y_new):] = y_new
            data = {
                'x': x.tolist(),
                'y': y.tolist()
            }
            
            socketio.emit('data', json.dumps(data))
            socketio.sleep(0.1)
        else:
            RUNNING = False
            break


@socketio.on('start')
def on_start():
    print('start')
    global RUNNING
    RUNNING = True
    socketio.start_background_task(generate_data)
    # socketio.spawn(generate_data)

@socketio.on('stop')
def on_stop():
    print('stop')
    global RUNNING
    RUNNING = False


if __name__ == '__main__':
    webbrowser.open('http://localhost:5000')
    socketio.run(app)
