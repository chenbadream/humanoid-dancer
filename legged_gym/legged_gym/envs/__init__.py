from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.h1.h1_config import H1Cfg, H1PPOCfg
from legged_gym.envs.h1.h1_env import H1Robot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "h1", H1Robot, H1Cfg(), H1PPOCfg())
