import onnxruntime
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# coco labels list
with open('labels/coco_classes.txt') as f:
    labels = f.readlines()
coco_labels = [c.strip() for c in labels]


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


# Resized image (1x3x416x416) Original image size (1x2) which is [image.size[1], image.size[0]]
def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data


def main():
    # Load image
    image = Image.open('images/input/roadsegmentation/25.jpeg')

    # Resized
    image_data = preprocess(image)
    image_size = np.array([image.size[1], image.size[0]], dtype='float32').reshape(1, 2)

    # Check
    # print(type(image_data))
    # print(image_data)

    '''
    The model has 3 outputs. boxes: (1x'n_candidates'x4), 
    the coordinates of all anchor boxes, scores: (1x80x'n_candidates'), 
    the scores of all anchor boxes per class, indices: ('nbox'x3), selected indices from the boxes tensor.
    The selected index format is (batch_index, class_index, box_index). 
    '''

    # 1. Make session
    session = onnxruntime.InferenceSession('models/yolov3.onnx')

    # 2. Get input/output name
    input_name = session.get_inputs()[0].name  # 'image'
    input_name_img_shape = session.get_inputs()[1].name  # 'image_shape'

    output_name_boxes = session.get_outputs()[0].name  # 'boxes'
    output_name_scores = session.get_outputs()[1].name  # 'scores'
    output_name_indices = session.get_outputs()[2].name  # 'indices'

    # 3. run
    outputs_index = session.run([output_name_boxes, output_name_scores, output_name_indices],
                                {input_name: image_data, input_name_img_shape: image_size})

    output_boxes = outputs_index[0]
    output_scores = outputs_index[1]
    output_indices = outputs_index[2]

    # Result
    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in output_indices:
        out_classes.append(idx_[1])
        out_scores.append(output_scores[tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(output_boxes[idx_1])

    # Make Figure and Axes
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    caption = []
    draw_box_p = []
    for i in range(0, len(out_classes)):
        box_xy = out_boxes[i]
        p1_y = box_xy[0]
        p1_x = box_xy[1]
        p2_y = box_xy[2]
        p2_x = box_xy[3]
        draw_box_p.append([p1_x, p1_y, p2_x, p2_y])
        draw = ImageDraw.Draw(image)
        # Draw Box
        draw.rectangle(draw_box_p[i], outline=(255, 0, 0), width=5)

        caption.append(coco_labels[out_classes[i]])
        caption.append('{:.2f}'.format(out_scores[i]))
        # Draw Class name and Score
        ax.text(p1_x, p1_y,
                ': '.join(caption),
                style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})

        caption.clear()

    # Output result image
    img = np.asarray(image)
    ax.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
