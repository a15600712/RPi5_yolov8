from ultralytics import YOLO
from pathlib import Path


# .pt files contains names in there but exported onnx/tflite don't have them.
yolo_default_label_names = {0: 'Monkey'}


class YoloDetector:
    def __init__(self, model_path, task='detect'):
        self.model = YOLO(model_path, task=task)
        self.imgsz = 640  # assume 640 at the moment since it is the default one

    def predict(self, frame, conf):
        return self.model.predict(source=frame, save=False, conf=conf, save_txt=False, show=False, verbose=False,
                                  imgsz=self.imgsz)

    def get_label_names(self):
        if self.model.names is None or len(self.model.names) == 0:
            return yolo_default_label_names
        return self.model.names



class YoloDetectorWrapper:
    def __init__(self, model_path):
        model_path = Path(model_path)
        self.detector = YoloDetector(model_path)

    def predict(self, frame, conf=0.75):
        return self.detector.predict(frame, conf=conf)

    def get_label_names(self):
        return self.detector.get_label_names()
