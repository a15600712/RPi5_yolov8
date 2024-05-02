import threading
import cv2
import numpy as np
from yolo_manager import YoloDetectorWrapper
from utils import draw_annotation, sendLineNotify
import time
from picamera2 import Picamera2


class VideoThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.should_run = True
        self.detect_frame = True
        self.frame_ready = threading.Event()

    def run(self):
        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(main={"size": (640, 640), "format": "RGB888"})
        picam2.configure(camera_config)
        picam2.start()

        while self.should_run:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame is not None and self.detect_frame:
                app.process_frame(frame)
                self.frame_ready.wait()  # Wait until frame is processed
                self.frame_ready.clear()
            else:
                time.sleep(0.0001)

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


class App:
    def __init__(self):
        self.yolo_detector = YoloDetectorWrapper("/home/anon/no gui/RPi5_yolov8/models/model_datasetv9.pt")
        self.lockerstatus = False
        target_indices = {0}  # Assuming target index for detection
        self.detection_counter = FrameCounter(target_indices, 2)

    def process_frame(self, cv_img):
        results = self.yolo_detector.predict(cv_img)
        if self.detection_counter.check_detection_results(results):
            if not self.lockerstatus:
                print("Lock All")
                sendLineNotify(draw_annotation(cv_img, self.yolo_detector.get_label_names(), results))
                self.lockerstatus = True
            else:
                print("All Locked")
                self.thread.should_run = False
                raise SystemExit('Lock engaged, stopping all operations')
        self.thread.frame_ready.set()  # Signal that frame has been processed


if __name__ == "__main__":
    app = App()
    app.thread = VideoThread()
    app.thread.start()

    # app.thread.stop()
    # app.thread.join()
