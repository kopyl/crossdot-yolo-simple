from flask import Flask, request, jsonify
from yolov8 import YOLOv8
import cv2
import numpy as np
import urllib.request
import os


app = Flask(__name__)

class_names = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
 }
model_path = os.environ.get("yolov8s-base.onnx")
yolov8_detector = YOLOv8(model_path, conf_thres=0.35, iou_thres=0.6)


def get_image_from_url(image_url):
    try:
        req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as url:
            image_contents = url.read()
    except urllib.error.HTTPError as e:
        return {"error": "HTTPError", "reason": "Was not able to download image."}
    except urllib.error.URLError as e:
        return {"error": "URLError", "reason": "Was not able to download image."}
    except ConnectionResetError:
        return {"error": "ConnectionResetError", "reason": "Was not able to download image."}

    arr = np.asarray(bytearray(image_contents), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return {"image": img}


def resize_with_pad(image,
                    new_shape,
                    padding_color = (255, 255, 255)):
    """
    https://gist.github.com/IdeaKing/11cf5e146d23c5bb219ba3508cca89ec
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image


def predict_single_image(img_url):
    img = get_image_from_url(img_url)
    if "error" in img:
        return img
    img = img["image"]

    img = resize_with_pad(img, (800, 800))
    boxes, scores, class_ids = yolov8_detector(img)
    prediction = map(
        lambda x:
            {
                "label": class_names[x[0]],
                "score": float(x[1]),
                "box": [float(x) for x in x[2]]
            },
        zip(class_ids, scores, boxes)
    )
    return list(prediction)


@app.route("/v0/classify/image", methods=["POST"])
def home():
    data = request.get_json()
    image_url = data.get("input")

    if not image_url:
        return jsonify({"result": "Image url not provided"}), 400
    print(image_url)

    prediction = predict_single_image(image_url)
    return jsonify(prediction), 200


@app.route("/", methods=["GET"])
def home_get():
    return "Hello World!"


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=80)
    # fastwsgi.run(wsgi_app=app, host='0.0.0.0', port=80)
    # app.run(debug=True, host="0.0.0.0", port=80, threaded=True)
    # app.run(debug=True, host="0.0.0.0", port=80, threaded=False, processes=64)
