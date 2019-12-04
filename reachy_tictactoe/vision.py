import numpy as np
import cv2 as cv
import os

from PIL import Image

from edgetpu.utils import dataset_utils
from edgetpu.classification.engine import ClassificationEngine


from .utils import piece2id

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, 'models')

boxes_classifier = ClassificationEngine(os.path.join(model_path, 'ttt-boxes.tflite'))
boxes_labels = dataset_utils.read_label_file(os.path.join(model_path, 'ttt-boxes.txt'))

# valid_classifier = ClassificationEngine('...')
# valid_labels = dataset_utils.read_label_file('...')


board_cases = np.array((
    ((345, 445, 340, 430),
     (445, 540, 340, 430),
     (540, 645, 340, 430),),

    ((330, 435, 430, 540),
     (435, 550, 430, 540),
     (550, 660, 430, 540),),

    ((310, 430, 540, 680),
     (430, 550, 540, 680),
     (550, 670, 540, 680),),
))


def get_board_configuration(img):
    board = np.zeros((3, 3), dtype=np.uint8)

    for row in range(3):
        for col in range(3):
            lx, rx, ly, ry = board_cases[row, col]
            piece, _ = identify_box(img[ly:ry, lx:rx])
            # We inverse the board to present it from the Human point of view
            board[2 - row, 2 - col] = piece2id[piece]

    return board


def identify_box(box_img):
    res = boxes_classifier.classify_with_image(img_as_pil(box_img), top_k=1)
    assert res

    label_index, score = res[0]
    label = boxes_labels[label_index]
    return label, score


def is_board_valid(img):
    return True
    res = valid_classifier.classify_with_image(img_as_pil(img), top_k=1)
    assert res

    label_index, score = res[0]
    label = valid_labels[label_index]

    return label == 'valid'


def img_as_pil(img):
    return Image.fromarray(cv.cvtColor(img.copy(), cv.COLOR_BGR2RGB))
