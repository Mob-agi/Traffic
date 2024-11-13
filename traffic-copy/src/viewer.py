import numpy as np
from highway_env.envs.common.graphics import EnvViewer


class FixEnvViewer(EnvViewer):
    def window_position(self) -> np.ndarray:
        return self.config.get("centering_position", [0, 0])
