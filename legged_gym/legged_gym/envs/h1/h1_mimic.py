from .h1_robot import H1Robot
from .h1_mimic_config import H1MimicCfg

class H1Mimic(H1Robot):
    def __init__(self, cfg: H1MimicCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)