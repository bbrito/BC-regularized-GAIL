import numpy as np
import gym

from gym import spaces
from gym.utils import seeding
from imitation.policies.base import HardCodedPolicy
from .lq_game import LQFeedbackGame, AffineStrategy
import imitating_games

class FeedbackGameEnv(gym.Env):
    """
    A generic, stateful wrapper that creates a `gym.Env` from a game that makes the player with
    index `protagonist_index` the protagonist, converting the problem to a single-player problem.
    """

    def __init__(self):
        game = imitating_games.GuidanceFeedbackGame()
        self.stage_strategies = game.solve()
        self.protagonist = game.players[0]#game.players[protagonist_index]
        self.game = game
        nx = game.dynamics.dims[0]
        nu_protagonist = len(self.protagonist.input_indices)
        self.action_space = spaces.Box(
            low=-np.ones(nu_protagonist) * np.inf,
            high=np.ones(nu_protagonist) * np.inf,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.ones(nx) * np.inf, high=np.ones(nx) * np.inf, dtype=np.float32
        )
        # The internal "episode memory" that also serves as state.
        self.seed()

    # Note: currently we don't do any seeding (because we always start from the same initial state)
    def seed(self, seed=None):
        # TODO: is this safe? Or will this correlate the rngs?
        self.np_random, main_seed = seeding.np_random(seed)
        [action_seed] = self.action_space.seed(main_seed + 1)
        [observation_seed] = self.observation_space.seed(main_seed + 2)
        return [main_seed, action_seed, observation_seed]

    def reset(self):
        x0 = self.np_random.uniform(low=-1, high=1, size=self.game.dynamics.dims[0])
        self.trajectory = [x0]
        return x0

    def step(self, action: np.ndarray):
        """
        Apply the `action` from the current state (end of `self.trajectory`) by overwriting the
        protagonist action.
        """
        t = len(self.trajectory) - 2
        x = self.trajectory[-1]
        u = self.stage_strategies[t].control_input(x).copy()
        # overwrite the protagonist actions
        u[self.protagonist.input_indices] = action
        next_x = self.game.dynamics.next_state(x, u)
        self.trajectory.append(next_x)

        ob = self.trajectory[-1]
        # Note: We reward the agent for the *next* state since the current state cannot be
        # influenced by the `action` and we terminate the game once we rolled out the full horizon.
        reward = -(self.protagonist.state_cost(next_x) + self.protagonist.input_cost(u))
        done = len(self.trajectory) >= self.game.horizon
        info = None


        return ob, reward, done, {}

    def render(self, mode="human"):
        return self.game.visualize(self.trajectory)


class GameSolverExpertPolicy(HardCodedPolicy):
    def __init__(self, game_env):
        super().__init__(game_env.observation_space, game_env.action_space)
        self.game_env = game_env

    def _choose_action(self, obs: np.ndarray):
        # TODO: This is not quite correct. Currently it just always takes the first action because
        # time is not part of the state yet.
        x = obs
        expert_strategy: AffineStrategy = self.game_env.stage_strategies[0]
        u_expert = expert_strategy.control_input(x)[
            self.game_env.protagonist.input_indices
        ]
        return u_expert
