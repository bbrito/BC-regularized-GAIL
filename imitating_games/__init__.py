from .example_guidance_game import GuidanceFeedbackGame
from .gym import FeedbackGameEnv, GameSolverExpertPolicy
from gym.envs.registration import register

register(
    id="FeedbackGame-v0",
    entry_point="imitating_games.gym:FeedbackGameEnv",
)
