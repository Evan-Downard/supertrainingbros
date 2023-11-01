import gymnasium as gym
import time
# import ale_py
# import sys
# input = sys.stdin.readline

# env = gym.make("Breakout-v4", render_mode='human')
# env = gym.make('MountainCar-v0', render_mode="human")
env = gym.make('ALE/MarioBros-v5', render_mode="human")

env.metadata['render_fps'] = 30


class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""

    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty
        import sys

    def __call__(self):
        import sys
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


getch = _Getch()

# Get the initial state
state = env.reset()

# Take an action
action = env.action_space.sample()

# Get the next state and reward
next_state, reward, done, truncated, info = env.step(action)

# Check if the game is over
if done:
    print('Game over!')

def run_game():
    max_actions = 100
    num_actions = 0
    next_state, reward, done, truncated, info = (0, 0, False, None, None)
    # Continue playing until the game is over
    while not done and num_actions < max_actions:
        # print(action)
        # Take an action
        num_actions += 1
        action = env.action_space.sample()

        # Get the next state and reward
        # env.step(action)
        next_state, reward, done, truncated, info = env.step(action)
        # print(reward, done, truncated, info)

        # Check if the game is over
        if done or num_actions > max_actions:
            env.close()
            print('Game over!')
            break

    env.close()
