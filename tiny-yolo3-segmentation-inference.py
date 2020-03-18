import numpy as np
from PIL import Image
import onnxruntime as rt
import onnx
from onnx import numpy_helper
import json
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import time


# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data


def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)


# -------------------- SEGMENTATION ---------------------
# image = Image.open('city1.png')
# # input
# image_data = preprocess(image)
# image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)
# labels = np.loadtxt('coco_classes.txt')
# -------------------------------------------------------

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def postprocess(result):
    return softmax(np.array(result)).tolist()


labels = load_labels('imagenet-simple-labels.json')
image = Image.open('images/dogm.jpg')
# image = Image.open('images/plane.jpg')

print("Image size: ", image.size)
plt.axis('off')
display_image = plt.imshow(image)
image_data = np.array(image).transpose(2, 0, 1)
input_data = preprocess(image_data)

# Run the model on the backend
session = rt.InferenceSession('models/resnet50v2/resnet50v2.onnx', None)
# get the name of the first input of the model
input_name = session.get_inputs()[0].name
print('Input Name:', input_name)
start = time.time()
raw_result = session.run([], {input_name: input_data})
end = time.time()
res = postprocess(raw_result)

inference_time = np.round((end - start) * 1000, 2)
idx = np.argmax(res)

print('========================================')
print('Final top prediction is: ' + labels[idx])
print('========================================')

print('========================================')
print('Inference time: ' + str(inference_time) + " ms")
print('========================================')

sort_idx = np.flip(np.squeeze(np.argsort(res)))
print('============ Top 5 labels are: ============================')
print(labels[sort_idx[:5]])
print('===========================================================')

plt.axis('off')
display_image = plt.imshow(image)

# out_boxes, out_scores, out_classes = [], [], []
# for idx_ in indices:
#     out_classes.append(idx_[1])
#     out_scores.append(scores[tuple(idx_)])
#     idx_1 = (idx_[0], idx_[2])
#     out_boxes.append(boxes[idx_1])
