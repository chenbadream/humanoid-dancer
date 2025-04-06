from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.h1 import h1_robot, h1_mimic, h1_config, h1_mimic_config
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( h1_config.H1Cfg.name, h1_robot.H1Robot, h1_config.H1Cfg(), h1_config.H1PPOCfg())
task_registry.register( h1_mimic_config.H1MimicCfg.name, h1_mimic.H1Mimic, h1_mimic_config.H1MimicCfg(), h1_mimic_config.H1MimicPPOCfg())
