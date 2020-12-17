import numpy as np
import cv2 as cv
import logging
import os

from PIL import Image

from edgetpu.utils import dataset_utils
from edgetpu.classification.engine import ClassificationEngine


from .utils import piece2id
from .detect_board import get_board_cases


logger = logging.getLogger('reachy.tictactoe')


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, 'models')

boxes_classifier = ClassificationEngine(os.path.join(model_path, 'ttt-boxes.tflite'))
boxes_labels = dataset_utils.read_label_file(os.path.join(model_path, 'ttt-boxes.txt'))

valid_classifier = ClassificationEngine(os.path.join(model_path, 'ttt-valid-board.tflite'))
valid_labels = dataset_utils.read_label_file(os.path.join(model_path, 'ttt-valid-board.txt'))


board_cases = np.array((
    ((209, 316, 253, 346), #Coordinates first board cases (top-left corner) (Xbl, Xbr, Ytr, Ybr)
     (316, 425, 253, 346), #Coordinates second board cases
     (425, 529, 253, 346),),

    ((189, 306, 346, 455),
     (306, 428, 346, 455),
     (428, 538, 346, 455),),

    ((174, 299, 455, 580),
     (299, 429, 455, 580),
     (429, 551, 455, 580),),
))

# left, right, top, bottom
board_rect = np.array((
    250, 700, 350, 1000,
))


def get_board_configuration(img):
    board = np.zeros((3, 3), dtype=np.uint8)

    # try:
    #     custom_board_cases = get_board_cases(img)
    # except Exception as e:
    #     logger.warning('Board detection failed', extra={'error': e})
    #     custom_board_cases = board_cases
    custom_board_cases = board_cases
    sanity_check = True

    for row in range(3):
        for col in range(3):
            lx, rx, ly, ry = custom_board_cases[row, col]
            piece, score = identify_box(img[ly:ry, lx:rx])
            #if score < 0.9:
            #    sanity_check = False
            #    return [], sanity_check
            # We invert the board to present it from the Human point of view
            if score < 0.9:
                piece = 0
            board[2 - row, 2 - col] = piece
    return board, sanity_check


def identify_box(box_img):

    res = boxes_classifier.classify_with_image(img_as_pil(box_img), top_k=1)
    assert res

    label, score = res[0]

    return label, score


def is_board_valid(img):
    lx, rx, ly, ry = board_rect
    board_img = img[ly:ry, lx:rx]
    res = valid_classifier.classify_with_image(img_as_pil(board_img), top_k=1)
    assert res

    label_index, score = res[0]
    label = valid_labels[label_index]

    logger.info('Board validity check', extra={
        'label': label,
        'score': score,
    })

    return label == 'valid' and score > 0.65


def img_as_pil(img):
    return Image.fromarray(cv.cvtColor(img.copy(), cv.COLOR_BGR2RGB))
