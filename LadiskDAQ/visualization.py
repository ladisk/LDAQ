import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QPushButton
from PyQt5.QtCore import QTimer, Qt
import numpy as np
import sys
import random
import time
import types

class Visualization:
    def __init__(self, layout):
        self.layout = layout
        

    def run(self, core):
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        with self.app:
            self.main_window = MainWindow(self.layout, core, self.app)
            self.main_window.show()
            self.app.exec_()
        
        core.is_running_global = False

class MainWindow(QMainWindow):
    def __init__(self, layout, core, app):
        super().__init__()
        self.app = app
        self.layout = layout
        self.core = core
        self.setWindowTitle('Data Acquisition and Visualization')
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout_widget = QVBoxLayout(self.central_widget)

        self.close_button = QPushButton('Close')
        self.close_button.clicked.connect(self.close_app)
        self.layout_widget.addWidget(self.close_button)

        self.init_plots()
        self.init_timer()

    
    def init_plots(self):
        self.time_start = time.time()
        self.plots = {}
        grid_layout = pg.GraphicsLayoutWidget()
        self.layout_widget.addWidget(grid_layout)
        self.subplots = {}

        for source, positions in self.layout.items():
            plot_channels = []
            for pos, channels in positions.items():
                if pos not in self.subplots.keys():
                    self.subplots[pos] = grid_layout.addPlot(*pos)


                for ch in channels:
                    if isinstance(ch, tuple):
                        x, y = ch
                        line = self.subplots[pos].plot(pen=pg.mkPen(color=random_color()))
                        plot_channels.append((line, x, y))
                    elif isinstance(ch, types.FunctionType):
                        print('function')
                    else:
                        line = self.subplots[pos].plot(pen=pg.mkPen(color=random_color()))
                        plot_channels.append((line, ch))

            self.plots[source] = plot_channels


    def init_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)


    def update_plots(self):
        if not self.core.is_running_global:
            self.close_app()

        new_data = self.core.get_measurement_dict(2)
        for source, plot_channels in self.plots.items():
            for line, *channels in plot_channels:
                if len(channels) == 1:
                    ch = channels[0]
                    line.setData(new_data[source]["time"], new_data[source]["data"][:, ch])
                else:
                    x_ch, y_ch = channels
                    x = new_data[source]['data'][:, x_ch]
                    y = new_data[source]['data'][:, y_ch]
                    line.setData(x, y)



    def close_app(self):
        self.timer.stop()
        self.app.quit()
        self.close()

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
