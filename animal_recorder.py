import os

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from yolo_manager import YoloDetectorWrapper
from utils import SimpleFPS, draw_fps, draw_annotation
import argparse
import time
from pathlib import Path
from picamera2 import Picamera2
import subprocess




class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, rotate_camera, rtmp_url):
        super().__init__()
        self.detect_frame = True
        self.should_run = True
        self.rotate_camera = rotate_camera

        self.rtmp_url = rtmp_url

    def run(self):
        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(main={"size":(640,640),"format":"RGB888"}, raw={"size": (640, 640)})
        # camera_config = picam2.create_still_configuration(main={"size": (640, 480)}, lores={"size": (640, 480)}, display="lores")
        picam2.configure(camera_config)
        picam2.start()

        video_broadcaster = None
        if self.rtmp_url is not None:
            print(f'streaming to {self.rtmp_url}')
            # video_broadcaster = VideoBroadcaster(rtmp_url=self.rtmp_url)

        while self.should_run:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            
            if frame is not None:

                if self.rotate_camera:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                if video_broadcaster is not None:
                    video_broadcaster.update(frame)
                
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
    def __init__(self, camera_test_only, rotate_camera, rtmp_url):
        super().__init__()

        self.camera_test_only = camera_test_only

        if camera_test_only:
            self.yolo_detector = None
        else:
            self.yolo_detector = YoloDetectorWrapper(args.model)

        target_indices = {0}  # Monkey
        # if we find targets in least 3 frames in a row, we start recording
        self.detection_counter = FrameCounter(target_indices, 3)

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
        self.thread = VideoThread(rotate_camera, rtmp_url)
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
                # self.thread.record_video() ##This line is for record but it can be edit to do anything
                print("Monkey Detected Lock the lock")
        # fps, _ = self.fps_util.get_fps()
        # draw_fps(display_img, fps)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/anon/Desktop/Mine_Project/RPi5_yolov8/models/nanomodel_dataset_v7.pt")
    parser.add_argument("--rtmp_url", type=str, default=None)
    
    parser.add_argument('--camera_test', action=argparse.BooleanOptionalAction)
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--rotate_camera', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    app = QApplication(sys.argv)
    a = App(camera_test_only=args.camera_test, rotate_camera=args.rotate_camera,
            rtmp_url=args.rtmp_url)
    a.show()
    sys.exit(app.exec_())
