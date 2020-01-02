import numpy as np
import logging
import time

from reachy import Reachy
from reachy.parts import RightArm, Head
from reachy.trajectory import TrajectoryPlayer

from .vision import get_board_configuration, is_board_valid
from .utils import piece2id, id2piece, piece2player
from .moves import moves, rest_pos, base_pos
from .rl_agent import value_actions
from . import behavior


logger = logging.getLogger('reachy.tictactoe')


class TictactoePlayground(object):
    def __init__(self):
        logger.info('Creating the playground')

        self.reachy = Reachy(
            right_arm=RightArm(
                luos_port='/dev/ttyUSB*',
                hand='force_gripper',
            ),
            head=Head(
                camera_id=0,
                luos_port='/dev/ttyUSB*',
            ),
        )

        self.pawn_played = 0

    def setup(self):
        logger.info('Setup the playground')
        self.reachy.head.homing()

        for antenna in self.reachy.head.motors:
            antenna.compliant = False
            antenna.goto(
                goal_position=0, duration=2,
                interpolation_mode='minjerk',
            )
        self.goto_rest_position(5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        logger.info(
            'Closing the playground',
            extra={
                'exc': exc,
            }
        )
        self.reachy.close()

    # Playground and game functions

    def reset(self):
        logger.info('Resetting the playground')

        self.pawn_played = 0
        empty_board = np.zeros((3, 3), dtype=np.uint8).flatten()

        return empty_board

    def is_ready(self, board):
        return np.sum(board) == 0

    def random_look(self):
        dy = 0.4
        y = np.random.rand() * dy - (dy / 2)

        dz = 0.75
        z = np.random.rand() * dz - 0.5

        self.reachy.head.look_at(0.5, y, z)
        time.sleep(1.5)

    def play_invite(self):
        look_pos = (
            ((0.5, 0, 0), 0.2),
            ((0.5, 0.2, -0.4), 0.5),
            ((0.5, 0, 0), 0.2),
            ((0.5, 0.2, -0.4), 0.5),
            ((0.5, 0, -0.55), 0.5),
            ((0.5, 0.2, -0.4), 0.5),
            ((0.5, 0, -0.55), 1),
            ((0.5, 0, 0), 1),
        )

        for (pos, dur) in look_pos:
            self.reachy.head.look_at(*pos)
            time.sleep(dur)

    def run_random_idle_behavior(self):
        logger.info('Reachy is playing a random idle behavior')
        r = np.random.rand()

        if r < 0.25:
            self.play_invite()
        elif r < 0.75:
            self.random_look()
        else:
            self.reachy.head.look_at(1, 0, 0)
            time.sleep(1)

    def coin_flip(self):
        coin = np.random.rand() > 0.5
        logger.info(
            'Coin flip',
            extra={
                'first player': 'reachy' if coin else 'human',
            },
        )
        return coin

    def analyze_board(self):
        # board_pos = (0.5, 0, -0.6)
        # self.reachy.head.look_at(*board_pos)

        for disk in self.reachy.head.neck.disks:
            disk.compliant = False
            disk.target_rot_speed = 50

        time.sleep(0.1)

        self.reachy.head.look_at(0.5, 0, -0.65)
        # Wait for stabilization
        time.sleep(1.25)

        img = self.reachy.head.get_image()

        if not is_board_valid(img):
            self.reachy.head.compliant = False
            time.sleep(0.1)
            self.reachy.head.look_at(1, 0, 0)
            time.sleep(2)
            return

        board = get_board_configuration(img)

        # TEMP:
        import os
        import cv2 as cv
        while True:
            i = np.random.randint(424242)
            path = f'/tmp/snap.{i}.jpg'
            if not os.path.exists(path):
                cv.imwrite(path, img)

                logger.info(
                    'Analyzing board',
                    extra={
                        'board': board,
                        'img_path': path,
                    },
                )

                break

        self.reachy.head.compliant = False
        time.sleep(0.1)
        self.reachy.head.look_at(1, 0, 0)
        time.sleep(2)

        return board.flatten()

    def cheating_detected(self, board, last_board):
        # last is just after the robot played
        delta = board - last_board

        # Nothing changed
        if np.all(delta == 0):
            return False

        # A single cube was added
        if len(np.where(delta == piece2id['cube'])[0]) == 1:
            return False

        # A single cylinder was added
        if len(np.where(delta == piece2id['cylinder'])[0]) == 1:
            return False

        logger.warning('Cheating detected', extra={
            'last_board': last_board,
            'current_board': board,
        })

        return True

    def shuffle_board(self):
        pass

    def choose_next_action(self, board):
        actions = value_actions(board)
        best_action, value = actions[0]

        logger.info(
            'Selecting Reachy next action',
            extra={
                'board': board,
                'actions': actions,
                'selected action': best_action,
            },
        )

        return best_action, value

    def play(self, action, actual_board):
        board = actual_board.copy()

        self.play_pawn(
            grab_index=self.pawn_played + 1,
            box_index=action + 1,
        )

        self.pawn_played += 1

        board[action] = piece2id['cylinder']

        logger.info(
            'Reachy playing pawn',
            extra={
                'board-before': actual_board,
                'board-after': board,
                'action': action + 1,
                'pawn_played': self.pawn_played + 1,
            },
        )

        return board

    def play_pawn(self, grab_index, box_index):
        self.reachy.head.look_at(
            0.3, -0.3, -0.5,
            duration=0.5,
        )

        # Goto base position
        self.goto_base_position()
        time.sleep(0.5)

        if grab_index >= 4:
            self.goto_position(
                moves['grab_3'],
                duration=1,
                wait=True,
            )

        # Grab the pawn at grab_index
        self.goto_position(
            moves[f'grab_{grab_index}'],
            duration=1,
            wait=True,
        )
        self.reachy.right_arm.hand.close()

        if grab_index >= 4:
            self.reachy.goto({
                'right_arm.shoulder_pitch': self.reachy.right_arm.shoulder_pitch.goal_position + 10,
                'right_arm.elbow_pitch': self.reachy.right_arm.elbow_pitch.goal_position - 30,
            }, duration=1,
               wait=True,
               interpolation_mode='minjerk',
               starting_point='goal_position',
            )

        # Lift it
        self.goto_position(
            moves['lift'],
            duration=1,
            wait=True,
        )

        self.reachy.head.look_at(0.3, 0, -0.5, duration=0.5)
        time.sleep(0.1)

        # Put it in box_index
        put = moves[f'put_{box_index}_smooth_10_kp']
        j = {
            m: j
            for j, m in zip(
                np.array(list(put.values()))[:, 0],
                list(put.keys())
            )
        }
        self.goto_position(j, duration=0.5, wait=True)
        TrajectoryPlayer(self.reachy, put).play(wait=True)

        self.reachy.right_arm.hand.open()

        # Go back to rest position
        self.goto_position(
            moves[f'back_{box_index}_upright'],
            duration=1,
            wait=True,
        )
        self.reachy.head.look_at(1, 0, 0, duration=1)

        if box_index in (8, 9):
            self.goto_position(
                moves[f'back_to_back'],
                duration=1,
                wait=True,
            )

        self.goto_position(
            moves[f'back_rest'],
            duration=2,
            wait=True,
        )
        self.goto_rest_position()

    def is_final(self, board):
        winner = self.get_winner(board)
        if winner in ('robot', 'human'):
            return True
        else:
            return 0 not in board

    def has_human_played(self, current_board, last_board):
        cube = piece2id['cube']

        return (
            np.any(current_board != last_board) or
            np.sum(current_board == cube) > np.sum(last_board == cube)
        )

    def get_winner(self, board):
        win_configurations = (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),

            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),

            (0, 4, 8),
            (2, 4, 6),
        )

        for c in win_configurations:
            trio = set(board[i] for i in c)
            for id in id2piece.keys():
                if trio == set([id]):
                    winner = piece2player[id2piece[id]]
                    if winner in ('robot', 'human'):
                        return winner

        return 'nobody'

    def run_celebration(self):
        logger.info('Reachy is playing its win behavior')
        behavior.happy(self.reachy)

    def run_draw_behavior(self):
        logger.info('Reachy is playing its draw behavior')
        behavior.surprise(self.reachy)

    def run_defeat_behavior(self):
        logger.info('Reachy is playing its defeat behavior')
        behavior.sad(self.reachy)

    # Robot lower-level control functions

    def goto_position(self, goal_positions, duration, wait):
        self.reachy.goto(
            goal_positions=goal_positions,
            duration=duration,
            wait=wait,
            interpolation_mode='minjerk',
            starting_point='goal_position',
        )

    def goto_base_position(self, duration=2.0):
        for m in self.reachy.right_arm.motors:
            m.compliant = False

        time.sleep(0.1)

        self.reachy.right_arm.shoulder_pitch.torque_limit = 75
        self.reachy.right_arm.elbow_pitch.torque_limit = 75
        time.sleep(0.1)

        self.goto_position(base_pos, duration, wait=True)

    def goto_rest_position(self, duration=2.0):
        self.goto_base_position(0.6 * duration)
        time.sleep(0.1)

        self.goto_position(rest_pos, 0.4 * duration, wait=True)
        time.sleep(0.1)

        self.reachy.right_arm.shoulder_pitch.torque_limit = 0
        self.reachy.right_arm.elbow_pitch.torque_limit = 0

        time.sleep(0.25)

        for m in self.reachy.right_arm.motors:
            if m.name != 'right_arm.shoulder_pitch':
                m.compliant = True

        time.sleep(0.25)

    def need_cooldown(self):
        motor_temperature = np.array([
            m.temperature for m in self.reachy.motors
        ])
        logger.info(
            'Checking Reachy motors temperature',
            extra={
                'temperatures': {
                    m.name: m.temperature for m in self.reachy.motors
                }
            }
        )
        return np.any(motor_temperature > 50)

    def wait_for_cooldown(self):
        self.goto_rest_position()

        while True:
            motor_temperature = np.array([
                m.temperature for m in self.reachy.motors
            ])

            logger.warning(
                'Motors cooling down...',
                extra={
                    'temperatures': {
                        m.name: m.temperature for m in self.reachy.motors
                    }
                },
            )

            if np.all(motor_temperature < 45):
                break

            time.sleep(30)
