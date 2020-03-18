import numpy as np
from PIL import Image
import onnxruntime as rt
import onnx
from onnx import numpy_helper
import json
import os
import glob
from PIL import Image, ImageDraw, ImageFont
import time


def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)


def preprocess(input_data):
    # convert the input data into the float32 input in case image is loaded with OpenCV (ubyte or 8-bit)
    img_data = input_data.astype('float32')

    # Subtract rbg mean to normalize or "center" the data (make it have 0 mean)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

    # adding a batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def postprocess(result):
    return softmax(np.array(result)).tolist()


labels = load_labels('labels/imagenet-simple-labels.json')
image = Image.open('images/input/imagenetclassification/dogm.jpg')

image_data = np.array(image).transpose(2, 0, 1)
input_data = preprocess(image_data)

# Run the model on the backend
loading_start = time.time()
session = rt.InferenceSession('models/resnet50v2/resnet50v2.onnx', None)
loading_end = time.time()
loading_time = np.round((loading_end - loading_start) * 1000, 2)

# get the name of the first input of the model
input_name = session.get_inputs()[0].name

start = time.time()
raw_result = session.run([], {input_name: input_data})
end = time.time()
res = postprocess(raw_result)

inference_time = np.round((end - start) * 1000, 2)
idx = np.argmax(res)
confidence = np.round(float(np.max(res, axis=0)), 2)

print('========================================')
print('Final top prediction is: ' + labels[idx])
print('Top prediction confidence is: ' + str(confidence))
print('========================================')

print('========================================')
print('Loading and model preparation took: ' + str(loading_time) + " ms")
print('Inference time: ' + str(inference_time) + " ms")
print('========================================')
