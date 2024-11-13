from src.dqn import DQNAgent
from src.env import DetachEnv


class TrafficSys:
    def __init__(
        self,
        render_mode: str,
        memory_size: int = 1000,
        batch_size: int = 32,
        target_update: int = 100,
        epsilon_decay: float = 5e-4,
    ):
        self.env = DetachEnv()
        self.render_mode = render_mode
        self.agent = DQNAgent(
            self.env,
            memory_size=memory_size,
            batch_size=batch_size,
            target_update=target_update,
            epsilon_decay=epsilon_decay,
            seed=0
        )

    def train(self, num_frames: int, save_path: str):
        self.env.render_mode = "none"
        self.agent.train(num_frames, save_path)
    
    def save(self, save_path: str):
        self.agent.save(save_path)
    
    def load(self, save_path: str):
        self.agent.load(save_path)
    
    def test(self):
        self.env.render_mode = "human"
        self.env.init_logger()
        self.agent.is_test = True
        env = self.env
        while True:
            obs, _ = self.env.reset()
            truncated = False
            while not truncated:
                action = self.agent.select_action(obs)
                obs, _, _, truncated, _ = self.env.step(action)
        env.close()