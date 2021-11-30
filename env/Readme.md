# The RetroArch Environment

I use the OpenAI Gym system for this as it makes implementing algorithms much easier.

## Files

In this repository you will find:

- retroarch.py - The main env file
- keys.py - Needed to allow python to input keystrokes
- windowcapture.py - Needed to use RetroArch as the observation
- windows.py - Used to find the name of open windows if you'd like to use another game

## RetroArch Gym Env

### Requirements

For windows only, other OS might just require some googling

```
pip install opencv-python numpy vgamepad matplotlib gym pytessy
```

vgamepad needs to install something on your computer but it is automatic. It asks everytime but you can cancel after it's done once.

I will have requirements files in each algorithm folder so this is more for information than use at this time.

### Setting up in Gym

Once you have gym installed you'll need to add the retroarch.py file to gym.

- navigate to your gym\envs\classic_control folder
- Place retroarch.py here
- Open the init.py file in classic_control
- Add the following line at the bottom
```
from gym.envs.classic_control.retroarch import RetroArch
```
- Go up to the gym\envs folder
- Open the init.py file in envs
- Add the following code at the bottom of the Classic section underneath Acrobat
```
register(
    id='RetroArch-v0',
    entry_point='gym.envs.classic_control:RetroArch',
)
```

### Env Explaination

The retroarch.py file has several sections to enable it to be a part of the gym ecosystem while keeping some flexibility for real time rl.

#### Variables

```
wincap = WindowCapture('RetroArch Gearboy 3.3.0') - Captures the Retroarch window
gamepad = vg.VX360Gamepad() - Allows python to input xbox controller button presses to Retroarch
keys = k.Keys() - Allows python to input key presses, useful for saving states or resetting Retroarch
ocrReader = PyTessy() - Text recognition, very fast, not very accurate
```

#### Buttons

This is how python presses and releases virtual xbox controller buttons
```
def A():
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    gamepad.update()
```

#### Regions Of Interest

This definitely isn't the best way of doing this, however, it works well enough.
This for example is used to read start on the menu screen.
```
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
```

#### RetroArch Class

the obs and action setup for the env
```
def __init__(self):
        super(RetroArch, self).__init__()
        # Define a 2-D observation space
        self.observation_space = spaces.Box(low = 0, 
                                            high = 255,
                                            shape=(200,200,3), dtype=np.uint8)
    
        
        # Define an action space ranging from 0 to 6
        self.action_space = spaces.Discrete(7,)
        self.state = 0
        self.seed()
```

#### Seed

Some algorithms require this

```
def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31

        return [seed1, seed2]
```

#### Reset

Resets the game which returns to the start menu, the menu is detected and the 'start' button is pressed

```
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

        self.steps = 0
        screen = wincap.get_screenshot()
        obs = cv.resize(screen, (200,200))
        return obs
```

#### Render

Not exactly useful in this case but its there is needed or wanted

```
def render(self, mode = "human"):
        assert mode in ["human"], "Invalid mode, must be \"human\""
        screen = wincap.get_screenshot()
        obs = cv.resize(screen, (200,200))
        if mode == "human":
            cv.imshow("Game", obs)
            cv.waitKey(1)
```

#### Close

Self explanatory?

```
def close(self):
        cv.destroyAllWindows() 
```

#### Get action meanings

Links action names with action numbers

```
def get_action_meanings(self):

        return {0: "NOOP", 1: "Right", 2: "Left", 3: "Down", 4: "Up", 5: "A", 6: "B"}
```

#### Clone/Restore State

Required by some algorithms, provides limited functionality in RetroArch as only one state can be saved at a time

```
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
```

#### Step

This is the main crux of the env, allows the agents to make actions and provides reward/penalties based on those actions.

- Score uses the ROI of the points on screen
- Episode sees the 'game over' message and resets the environment
- Actions define numbers for each of the gameboy buttons

```
def step(self, action):

        done = False
        self.steps += 1

        # obs
        screen = wincap.get_screenshot()
        obs = cv.resize(screen, (200,200))
        
        # Assert that it is a valid action 
        #assert self.action_space.contains(action), "Invalid Action"
        
        ### SCORE ###
        reward = 0
        try:
            score = roi_1()
            while True:
                reward = int(score) / 100
                break
        except:
            reward += 0

        ### EPISODES ###
        # If dead, end the episode.
        game = roi_3()
        try:
            while True:
                if 'GHAC' in game:
                    reward += 0
                    done = True
                    break
                else:
                    reward += 0
                    break
        except:
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
```
