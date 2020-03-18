#!/usr/bin/env python
# coding: utf-8

# # Inference demo for DUC models
# 
# ## Overview
# This notebook can be used for inference on DUC ONNX models. The demo shows how to use the trained models to do inference in MXNet.
# 
# ## Models supported
# * ResNet101_DUC_HDC
# 
# ## Prerequisites
# The following packages need to be installed before proceeding:
# * Protobuf compiler - `sudo apt-get install protobuf-compiler libprotoc-dev` (required for ONNX. This will work for any linux system. For detailed installation guidelines head over to [ONNX documentation](https://github.com/onnx/onnx#installation))
# * ONNX - `pip install onnx`
# * MXNet - `pip install mxnet-cu90mkl --pre -U` (tested on this version GPU, can use other versions. `--pre` indicates a pre build of MXNet which is required here for ONNX version compatibility. `-U` uninstalls any existing MXNet version allowing for a clean install)
# * numpy - `pip install numpy`
# * OpenCV - `pip install opencv-python`
# * PIL - `pip install pillow`
# 
# Also the following script (included in the repo) must be present in the same folder as this notebook:
# * `cityscapes_labels.py` (contains segmentation category labels)
#

import cv2 as cv
import numpy as np
from PIL import Image
import math
import onnxruntime
import cityscapes_labels
import time


def generate_labels_txt(labels_file_path):
    with open(labels_file_path) as f:
        labels = f.readlines()
    labels = [c.strip() for c in labels]
    return labels


# ### Preprocess image
# `preprocess()` : Prepares input image, subtracts RGB mean and converts it to ndarray to input to the model
def preprocess(im):
    # Convert to float32 to make it uniform
    test_img = im.astype(np.float32)
    # Extrapolate image with a small border in order obtain an accurate reshaped image after DUC layer
    test_shape = [im.shape[0], im.shape[1]]
    cell_shapes = [math.ceil(l / 8) * 8 for l in test_shape]
    test_img = cv.copyMakeBorder(test_img, 0, max(0, int(cell_shapes[0]) - im.shape[0]), 0,
                                 max(0, int(cell_shapes[1]) - im.shape[1]), cv.BORDER_CONSTANT, value=rgb_mean)
    # Just swap the position for image dimension and number of colors:
    # (width, height, rgb) -> (rgb, width, height), in this case: (800, 800, 3) -> (3, 800, 800)
    test_img = np.transpose(test_img, (2, 0, 1))
    # Subtract rbg mean to normalize or "center" the data (make it have 0 mean)
    for i in range(3):
        test_img[i] -= rgb_mean[i]

    # Ad additional dimension to numpy array, now we have (1, 3, 800, 800)
    test_img = np.expand_dims(test_img, axis=0)

    return test_img


# ### Generate predictions
# `get_palette()` : Returns predefined color palette for generating output segmentation map
# 
# `colorize()` : Generate the segmentation map using output `labels` generated by the model
# and color palette from `get_palette()`
# 
# `predict()` : Performs forward pass on the model using the preprocessed input,
# reshapes the output to match input image dimensions, generates colorized segmentation map using `colorize()`
def get_palette():
    # get train id to color mappings from file
    trainId2colors = {label.trainId: label.color for label in cityscapes_labels.labels}
    # prepare and return palette
    palette = [0] * 256 * 3
    for trainId in trainId2colors:
        colors = trainId2colors[trainId]
        if trainId == 255:
            colors = (0, 0, 0)
        for i in range(3):
            palette[trainId * 3 + i] = colors[i]
    return palette


def colorize(labels):
    # generate colorized image from output labels and color palette
    result_img = Image.fromarray(labels).convert('P')
    result_img.putpalette(get_palette())
    return np.array(result_img.convert('RGB'))


def postprocess(labels):
    '''
    Postprocessing function for DUC
    input : output labels from the network as numpy array, input image shape, desired output image shape
    output : confidence score, segmented image, blended image, raw segmentation labels
    '''
    ds_rate = 8
    label_num = 19
    cell_width = 2
    img_height, img_width = (800, 800)
    result_height, result_width = result_shape

    # re-arrange output
    test_width = int((int(img_width) / ds_rate) * ds_rate)
    test_height = int((int(img_height) / ds_rate) * ds_rate)
    feat_width = int(test_width / ds_rate)
    feat_height = int(test_height / ds_rate)
    labels = labels.reshape((label_num, 4, 4, feat_height, feat_width))
    labels = np.transpose(labels, (0, 3, 1, 4, 2))
    labels = labels.reshape((label_num, int(test_height / cell_width), int(test_width / cell_width)))

    labels = labels[:, :int(img_height / cell_width), :int(img_width / cell_width)]
    labels = np.transpose(labels, [1, 2, 0])
    labels = cv.resize(labels, (result_width, result_height), interpolation=cv.INTER_LINEAR)
    labels = np.transpose(labels, [2, 0, 1])

    # get softmax output
    softmax = labels

    # get classification labels
    results = np.argmax(labels, axis=0).astype(np.uint8)
    raw_labels = results

    # compute confidence score
    confidence = float(np.max(softmax, axis=0).mean())

    # generate segmented image
    result_img = Image.fromarray(colorize(raw_labels)).resize(result_shape[::-1])

    # generate blended image
    blended_img = Image.fromarray(cv.addWeighted(im[:, :, ::-1], 0.5, np.array(result_img), 0.5, 0))

    return confidence, result_img, blended_img, raw_labels


# read image as rgb
original_image_name = '25'
original_image_extension = '.jpeg'
original_image_path = 'images/input/roadsegmentation/' + original_image_name + original_image_extension
im = cv.imread(original_image_path)[:, :, ::-1]
# Firstly, resize image
im = cv.resize(im, (800, 800), interpolation=cv.INTER_AREA)
# set output shape (same as input shape)
result_shape = [im.shape[0], im.shape[1]]
# set rgb mean of input image (used in mean subtraction)
rgb_mean = cv.mean(im)
# display input image
pre = preprocess(im)
image_shape = [pre.shape[0], pre.shape[1]]

# ### Prepare ONNX model and set context
loading_start = time.time()
session = onnxruntime.InferenceSession('models/ResNet101_DUC_HDC.onnx', None)
loading_end = time.time()
loading_time = np.round((loading_end - loading_start) * 1000, 2)
print('========================================')
print('Loading and preparation of model took: ' + str(loading_time) + " ms")
print('========================================')
# get the name of the first input of the model
input_name = session.get_inputs()[0].name

# ### Get predictions
# conf, result_img, blended_img, raw = predict(pre)
inference_start = time.time()
raw_result = session.run([], {input_name: pre})
inference_end = time.time()
inference_time = np.round((inference_end - inference_start) * 1000, 2)
print('========================================')
print('Inference time: ' + str(inference_time) + " ms")
print('========================================')

# ### Display segmented output
# Each pixel is colored with a different color according to the class into which they are classified.
# Refer to cityscape_labels.py for category names and their corresponding colors.
(confidence, result_img, blended_img, raw_labels) = postprocess(raw_result[0].squeeze())

# ------------------------- Display segmented output -------------------------
# Each pixel is colored with a different color according to the class into which they are classified.
# Refer to cityscape_labels.py for category names and their corresponding colors.
result_img = result_img.save('images/output/roadsegmentation/' + 'onnxtest4' + '-result.png')

# ------------------------- Display blended output -------------------------
# The segmentation map is overlayed on top of the input image to have a more precise visualization
blended_img = blended_img.save('images/output/roadsegmentation/' + 'onnxtest4' + '-blended.png')

# ------------------------- Print confidence score -------------------------
# Confidence score is the maximum value of softmax output averaged over all pixels.
# The values lie in [0,1] with a higher value indicating that the model is more confident in
# classification of the pixels, which leads to a better output.
print('Confidence = %f' % confidence)
