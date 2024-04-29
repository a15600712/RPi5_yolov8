import cv2
import time
from ultralytics.utils.plotting import Annotator
import requests
import time
from PIL import Image
from io import BytesIO
import numpy as np

class SimpleFPS:
    def __init__(self):
        self.start_time = time.time()
        self.display_time_sec = 1  # update fps display
        self.fps = 0
        self.frame_counter = 0
        self.is_fps_updated = False

    def get_fps(self):
        elapsed = time.time() - self.start_time
        self.frame_counter += 1
        is_fps_updated = False

        if elapsed > self.display_time_sec:
            self.fps = self.frame_counter / elapsed
            self.frame_counter = 0
            self.start_time = time.time()
            is_fps_updated = True

        return int(self.fps), is_fps_updated


def draw_annotation(img, label_names, results):
    annotator = None
    for r in results:
        annotator = Annotator(img)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, label_names[int(c)])

    if annotator is not None:
        annotated_img = annotator.result()
    else:
        annotated_img = img.copy()

    return annotated_img

def sendLineNotify(image_array):
    t = time.time()
    t1 = time.localtime(t)
    now = time.strftime('%Y/%m/%d %H:%M:%S', t1)
    url = 'https://notify-api.line.me/api/notify'
    token = 'TokenHere'
    headers = {
      'Authorization': 'Bearer ' + token
    }
    data = {
      'message': now
    }
    image_array=cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')

    # Save the image to a BytesIO object
    image_file = BytesIO()
    image.save(image_file, format='JPEG')
    image_file.seek(0)  # Move the cursor to the start of the file

    # Create a 'files' dictionary to hold the file data
    files = {'imageFile': image_file}

    # Send POST request
    requests.post(url, headers=headers, data=data, files=files)
    image_file.close()  # Close the BytesIO object