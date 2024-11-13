from src.env import DetachEnv
from src.sys import TrafficSys

if __name__ == "__main__":

    num_frames = 10000
    
    is_train = False
    render_mode = "human"
    save_path = "logs/model.pth"

    sys = TrafficSys(render_mode)
    if is_train:
        sys.train(num_frames, save_path)
    else:
        sys.load(save_path)
        sys.test()
