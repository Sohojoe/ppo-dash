import gym
import cv2
from gym import spaces


class OTCPreprocessing(gym.Wrapper):
    """A class implementing image preprocessing for OTC agents.

    Specifically, this converts observations to greyscale. It doesn't
    do anything else to the environment.
    """

    def __init__(self, environment, custom_action_set=None):
        """Constructor for an Obstacle Tower preprocessor.

        Args:
            environment: Gym environment whose observations are preprocessed.

        """
        self.env = environment

        environment.action_meanings = ['NOOP']
        # environment.np_random, seed = gym.utils.seeding.np_random(None)

        self.game_over = False
        self.lives = 0  # Will need to be set by reset().
        if custom_action_set:
            self._action_lookup = custom_action_set
        else:
            self._action_lookup = {
                0: [0, 0, 0, 0],  # nop
                1: [1, 0, 0, 0],  # forward
                2: [2, 0, 0, 0],  # backward
                3: [0, 1, 0, 0],  # cam left
                4: [0, 2, 0, 0],  # cam right
                5: [1, 0, 1, 0],   # jump forward
                6: [1, 1, 0, 0],  # forward + cam left
                7: [1, 2, 0, 0]  # forward + cam right
            }

    # @property
    # def _action_lookup(self):
    #     return  {
    #         0: [0, 0, 0, 0],  # nop
    #         1: [1, 0, 0, 0],  # forward
    #         2: [2, 0, 0, 0],  # backward
    #         3: [0, 1, 0, 0],  # cam left
    #         4: [0, 2, 0, 0],  # cam right
    #         5: [1, 0, 1, 0],   # jump forward
    #         6: [1, 1, 0, 0],  # forward + cam left
    #         7: [1, 2, 0, 0],  # forward + cam right
    #     }

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return spaces.Discrete(len(self._action_lookup))
        # return self.env.action_space

    @property
    def reward_range(self):
        return self.env.reward_range

    @property
    def metadata(self):
        return self.env.metadata

    def reset(self):
        """Resets the environment. Converts the observation to greyscale, 
        if it is not. 

        Returns:
        observation: numpy array, the initial observation emitted by the
            environment.
        """
        observation = self.env.reset()
        # if(len(observation.shape)> 2):
        #     observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        return observation

    def render(self, mode):
        """Renders the current screen, before preprocessing.

        This calls the Gym API's render() method.

        Args:
        mode: Mode argument for the environment's render() method.
            Valid values (str) are:
            'rgb_array': returns the raw ALE image.
            'human': renders to display via the Gym renderer.

        Returns:
        if mode='rgb_array': numpy array, the most recent screen.
        if mode='human': bool, whether the rendering was successful.
        """
        return self.env.render(mode)

    def step(self, actionInput):
        """Applies the given action in the environment. Converts the observation to 
        greyscale, if it is not. 

        Remarks:

        * If a terminal state (from life loss or episode end) is reached, this may
            execute fewer than self.frame_skip steps in the environment.
        * Furthermore, in this case the returned observation may not contain valid
            image data and should be ignored.

        Args:
        action: The action to be executed.

        Returns:
        observation: numpy array, the observation following the action.
        reward: float, the reward following the action.
        is_terminal: bool, whether the environment has reached a terminal state.
            This is true when a life is lost and terminal_on_life_loss, or when the
            episode is over.
        info: Gym API's info data structure.
        """
        # ['Movement Forward/Back', 'Camera', 'Jump', 'Movement Left/Right']
        # [3, 3, 2, 3]
        from typing import Iterable
        if isinstance(actionInput, Iterable):
            action = self._action_lookup[actionInput[0]]
        else:
            action = self._action_lookup[actionInput]

        observation, reward, game_over, info = self.env.step(action)
        self.game_over = game_over
        # if(len(observation.shape)> 2):
        #     observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        info['actual_action']=actionInput
        info['actual_inner_action']=action
        return observation, reward, game_over, info

    def unwrap(self):
        if hasattr(self.env, "unwrapped"):
            return env.unwrapped
        elif hasattr(self.env, "env"):
            return unwrap(self.env.env)
        elif hasattr(self.env, "leg_env"):
            return unwrap(self.env.leg_env)
        else:
            return self.env
