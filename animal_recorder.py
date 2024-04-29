import os

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from yolo_manager import YoloDetectorWrapper
from utils import SimpleFPS, draw_annotation, sendLineNotify
import time
from picamera2 import Picamera2




class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.detect_frame = True
        self.should_run = True

    def run(self):
        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(main={"size":(640,640),"format":"RGB888"}, raw={"size": (640, 640)})
        picam2.configure(camera_config)
        picam2.start()

        while self.should_run:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if frame is not None:
                if self.detect_frame:
                    self.change_pixmap_signal.emit(frame)
                    self.detect_frame = False
            else:
                time.sleep(0.0001)

        print('VideoThread finished!')

    def stop(self):
        self.should_run = False



class FrameCounter:
    def __init__(self, detection_target_indices, num_frames):
        self.num_frames = num_frames
        self.detection_target_indices = detection_target_indices
        self.counter = 0

    def check_detection_results(self, detection_results):

        target_found = False
        for detection_result in detection_results:
            
            if len(detection_result.boxes) > 0:
                cls_id = int(detection_result.boxes.cls[0])
                if cls_id in self.detection_target_indices:
                    target_found = True
                    break

        if target_found:
            self.counter += 1
        else:
            self.counter = 0

        return self.counter >= self.num_frames


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.yolo_detector = YoloDetectorWrapper("/home/anon/Desktop/Mine_Project/RPi5_yolov8/models/nano_dataset_v8.pt")
        self.lockerstatus = False
        target_indices = {0}  # Monkey
        # if we find targets in least 2 frames in a row, we start recording
        self.detection_counter = FrameCounter(target_indices, 2)

        self.setWindowTitle("Monkey Detector")
        self.disply_width = 640
        self.display_height = 640
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

        self.fps_util = SimpleFPS()

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""

        if self.yolo_detector is None:
            display_img = cv_img
        else:
            results = self.yolo_detector.predict(cv_img)
            display_img = draw_annotation(cv_img, self.yolo_detector.get_label_names(), results)
            if self.detection_counter.check_detection_results(results):
                if(self.lockerstatus==False):
                    print("Lock\r\n")
                    sendLineNotify(display_img)
                    self.lockerstatus=True
                else:
                    print("Already Locked\r\n")


        qt_img = self.convert_cv_qt(display_img)
        self.image_label.setPixmap(qt_img)
        self.thread.detect_frame = True

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        print('closeEvent')
        self.thread.stop()
        self.thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication([])
    a = App()
    a.show()
    sys.exit(app.exec_())
