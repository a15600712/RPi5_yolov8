import cv2
import time
from ultralytics.utils.plotting import Annotator
import requests
import time
from PIL import Image
from io import BytesIO
import os

def draw_annotation(img, label_names, results):
    annotator = Annotator(img)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, label_names[int(c)])
    annotated_img = annotator.result() if annotator else img.copy()
    return annotated_img

def sendLineNotify(image_array):

    try:
        t = time.time()
        t1 = time.localtime(t)
        now = time.strftime('%Y/%m/%d %H:%M:%S', t1)
        url = 'https://notify-api.line.me/api/notify'
        token = 'HxHmQgU8pNo5438asyZzMKRI7O6W5X2xRt0KEVYzQW1'
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
    except Exception as e:
        print("Failed to send notification:", e)
    finally:
        image_file.close()