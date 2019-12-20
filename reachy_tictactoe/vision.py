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
    ((330, 430, 370, 455),
     (430, 520, 370, 455),
     (520, 620, 370, 455),),

    ((310, 415, 455, 560),
     (415, 520, 455, 560),
     (520, 635, 455, 560),),

    ((285, 410, 560, 700),
     (410, 530, 560, 700),
     (530, 655, 560, 700),),
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

    return label == 'valid' and score > 0.75


def img_as_pil(img):
    return Image.fromarray(cv.cvtColor(img.copy(), cv.COLOR_BGR2RGB))
