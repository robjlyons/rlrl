import cv2 as cv
import numpy as np
import time
import vgamepad as vg
import keys as k
import matplotlib.pyplot as plt
import gym
import pickle
import re
import multiprocessing

from windowcapture import WindowCapture
from gym import Env, spaces
from pytessy.pytessy import PyTessy
from PIL import ImageFilter, Image

from gym import utils
from gym.utils import seeding

wincap = WindowCapture('RetroArch Gearboy 3.3.0')
gamepad = vg.VX360Gamepad()
keys = k.Keys()
ocrReader = PyTessy()

def A():
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    gamepad.update()

def B():
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    gamepad.update()

def DPAD_UP():
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
    gamepad.update()

def DPAD_DOWN():
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
    gamepad.update()

def DPAD_LEFT():
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
    gamepad.update()

def DPAD_RIGHT():
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
    gamepad.update()

def START():
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
    gamepad.update()

def BACK():
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
    gamepad.update()

def NOOP():
    gamepad.reset()
    gamepad.update()

############# Regions Of Interest #############

def roi_1(): # Menu ROI for detecting start (START)
    screen = wincap.get_screenshot()
    menu = screen[335:355, 146:261] # [y1:y2, x1:x2]
    gray = get_grayscale(menu)
    resize = cv.resize(gray, (200,100))
    img = opening(resize)
    ocr = Image.fromarray(img)
    ocr = ocr.filter(ImageFilter.SHARPEN)
    imgBytes = ocr.tobytes()
    bytesPerPixel = int(len(imgBytes) / (ocr.width * ocr.height))
    result = ocrReader.read(ocr.tobytes(), ocr.width, ocr.height, bytesPerPixel, raw=True, resolution=600)
    roi_1 = str(result)

    return roi_1

def roi_2(): # Score Numbers
    screen = wincap.get_screenshot()
    screen = screen[45:68, 1:95] # [y1:y2, x1:x2]
    gray = get_grayscale(screen)
    resize = cv.resize(gray, (200,75))
    img = opening(resize)
    ocr = Image.fromarray(img)
    ocr = ocr.filter(ImageFilter.SHARPEN)
    imgBytes = ocr.tobytes()
    bytesPerPixel = int(len(imgBytes) / (ocr.width * ocr.height))
    ocr_result = ocrReader.read(ocr.tobytes(), ocr.width, ocr.height, bytesPerPixel, raw=True, resolution=600)
    roi_2 = str(ocr_result)

    return roi_2

def roi_3(): # Game Over ROI (GHAC)
    screen = wincap.get_screenshot()
    screen = screen[216:234, 121:213] # [y1:y2, x1:x2]
    gray = get_grayscale(screen)
    resize = cv.resize(gray, (200,75))
    img = opening(resize)
    ocr = Image.fromarray(img)
    ocr = ocr.filter(ImageFilter.SHARPEN)
    imgBytes = ocr.tobytes()
    bytesPerPixel = int(len(imgBytes) / (ocr.width * ocr.height))
    result = ocrReader.read(ocr.tobytes(), ocr.width, ocr.height, bytesPerPixel, raw=True, resolution=600)
    roi_3 = str(result)

    return roi_3

def roi_4(): # Lives
    screen = wincap.get_screenshot()
    screen = screen[23:45, 169:192] # [y1:y2, x1:x2]
    gray = get_grayscale(screen)
    resize = cv.resize(gray, (200,200))
    img = opening(resize)
    ocr = Image.fromarray(img)
    ocr = ocr.filter(ImageFilter.SHARPEN)
    imgBytes = ocr.tobytes()
    bytesPerPixel = int(len(imgBytes) / (ocr.width * ocr.height))
    ocr_result = ocrReader.read(ocr.tobytes(), ocr.width, ocr.height, bytesPerPixel, raw=True, resolution=600)
    roi_4 = str(ocr_result)

    return roi_4

def get_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

def replace_chars(text):
    """
    Replaces all characters instead of numbers from 'text'.
    
    :param text: Text string to be filtered
    :return: Resulting number
    """
    list_of_numbers = re.findall(r'\d+', text)
    #result_number = ''.join(list_of_numbers)
    return list_of_numbers

class RetroArch(Env):

    def __init__(self):
        super(RetroArch, self).__init__()
        # Define a 2-D observation space
        self.observation_space = spaces.Box(low = 0, 
                                            high = 255,
                                            shape=(200,200,3), dtype=np.uint8)
    
        
        # Define an action space ranging from 0 to 4
        self.action_space = spaces.Discrete(7,)
        self.state = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31

        return [seed1, seed2]

    def reset(self):

        # Restart RetroArch game
        keys.directKey("H")
        time.sleep(0.1)
        keys.directKey("H", keys.key_release)
        time.sleep(0.4)

        menu = roi_1()
        try:
            while True:
                if 'START' in menu:
                    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
                    gamepad.update()
                    time.sleep(0.4)
                    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
                    gamepad.update()
                    time.sleep(0.4)
                    break
                else:
                    print("Error")
                    break
        except:
            print("Error")

        gamepad.reset()
        gamepad.update()
        self.score_reward_old = 0
        screen = wincap.get_screenshot()
        obs = cv.resize(screen, (200,200))
        return obs

    def render(self, mode = "human"):
        assert mode in ["human"], "Invalid mode, must be \"human\""
        screen = wincap.get_screenshot()
        obs = cv.resize(screen, (200,200))
        if mode == "human":
            cv.imshow("Game", obs)
            cv.waitKey(1)
    
    def close(self):
        cv.destroyAllWindows()   

    def get_action_meanings(self):

        return {0: "NOOP", 1: "Right", 2: "Left", 3: "Down", 4: "Up", 5: "A", 6: "B"}

    def clone_state(self):
        # Save game
        self.state += 1
        keys.directKey("F2")
        time.sleep(0.1)
        keys.directKey("F2", keys.key_release)
        time.sleep(0.1)
        return self.state

    def restore_state(self, state):
        # Load last save
        keys.directKey("F4")
        time.sleep(0.1)
        keys.directKey("F4", keys.key_release)
        time.sleep(0.1)

    # Uses loss of life as terminal signal
    def train(self):
        self.training = False

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def step(self, action):
        # Flag that marks the termination of an episode
        done = False

        # obs
        screen = wincap.get_screenshot()
        obs = cv.resize(screen, (200,200))
        
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

        ### Reward ###
        reward = 0
        
        ### SCORE ###
        try:
            score = roi_2()
            score_reward = int(score)
            if score_reward > self.score_reward_old:
                self.score_reward_old = score_reward
                reward += self.score_reward_old
            else:
                reward += 0
        except:
            reward += 0

        ### LIVES ###
        lives = roi_4()
        if "1" in lives:
            reward += 0
            done = True
        else:
            reward += 0

        # Actions
        if action == 0:
            NOOP()
        elif action == 1:
            DPAD_RIGHT()
        elif action == 2:
            DPAD_LEFT()
        elif action == 3:
            DPAD_DOWN()
        elif action == 4:
            DPAD_UP()
        elif action == 5:
            A()
        elif action == 6:
            B()

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return obs, reward, done, info
