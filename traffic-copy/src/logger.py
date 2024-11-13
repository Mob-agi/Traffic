import pandas as pd
from highway_env.envs.common.abstract import AbstractEnv


class Logger:
    def __init__(self, log_dir: str, env: AbstractEnv):
        self.env = env
        self.log_dir = log_dir
        self.count_history = [False] * len(env.road.vehicles)
        header = ["frame"]
        for i in range(len(env.road.vehicles)):
            header.append(f"vehicle_{i}_velocity")
        self.data_frame = pd.DataFrame(columns=header)
        self.frame = 1

    def log_velocity(self):
        info = {"frame": self.frame}
        self.frame += 1
        for i in range(len(self.env.road.vehicles)):
            info[f"vehicle_{i}_velocity"] = (self.env.road.vehicles[i].velocity[0] ** 2 + self.env.road.vehicles[i].velocity[1] ** 2) ** 0.5
        self.frame += 1
        self.data_frame = pd.concat([self.data_frame, pd.DataFrame([info])])

    def log_final(self):
        avarage_values = self.data_frame.mean()
        info = {"frame": "average"}
        for column in self.data_frame.columns:
            if (column == "frame"):
                continue
            info[column] = avarage_values[column]
        self.data_frame = pd.concat([self.data_frame, pd.DataFrame([info])])

        num_of_controlled_vehicles = 0
        num_of_noncontrolled_vehicles = 0
        for i in range(len(self.count_history)):
            if self.count_history[i]:
                if self.env.road.vehicles[i] in self.env.controlled_vehicles:
                    num_of_controlled_vehicles += 1
                else:
                    num_of_noncontrolled_vehicles += 1
        success_num = 0
        for vehicle in self.env.controlled_vehicles:
            if not vehicle.crashed:
                success_num += 1
        print(
            f"The number of controlled vehicles passing through the main road: {len(self.env.controlled_vehicles) - num_of_controlled_vehicles}"
        )
        print(
            f"The number of non-controlled vehicles passing through the main road: {len(self.env.road.vehicles) - len(self.env.controlled_vehicles) - num_of_noncontrolled_vehicles}"
        )
        print(f"The number of controlled vehicles passing through the detached road: {num_of_controlled_vehicles}")
        print(f"The number of non-controlled vehicles passing through the main road: {num_of_noncontrolled_vehicles}")
        print(f"The number of controlled vehicles that have not crashed: {success_num}")
        
        self.data_frame.to_csv(f"{self.log_dir}/log.csv", index=False)

    def log_vehicle_num(self, road_index):
        for i in range(len(self.env.road.vehicles)):
            if self.env.road.vehicles[i].lane_index == road_index:
                self.count_history[i] = True
