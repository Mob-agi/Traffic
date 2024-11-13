import random
from typing import Dict

import highway_env
import numpy as np
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from src.logger import Logger
from src.vehicle import CustomControlledVehicle, RandomVehicle
from src.viewer import FixEnvViewer


class DetachEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> Dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                    },
                },
                # 奖励函数系数
                "speed_reward_w": 1.0,
                "acceleration_cost_w": -10.0,
                "crash_cost_w": -10.0,
                "lane_change_cost_w": -5.0,
                "export_reward_w": 20.0,
                # 车辆总数、被控车辆数量、进入匝道的被控车辆比例
                "vehicles_count": 10,
                "controlled_vehicles": 4,
                "exit_controlled_vehicles_ratio": 0.5,
                # 离开匝道的位置起点，距离起点的距离
                "export_of_cav": 300,
                "export_length": 100,
                "initial_lane_id": 0,
                "ego_spacing": 2,
                "vehicles_density": 1,
                "offroad_terminal": True,
                "duration": 40,
                "centering_position": [-0.0, 0.4],
                "screen_width": 2400,
                "screen_height": 300,
                "scaling": 2.5,
            }
        )
        return cfg

    def __init__(self, config=None, render_mode=None):
        super().__init__(config, render_mode)
        self.viewer = FixEnvViewer(self)
        self.logger = None
        
    def init_logger(self):
        self.logger = Logger("logs", self)

    def step(self, action):
        action = list(action)
        for index, vehicle in enumerate(self.controlled_vehicles):
            # 位于第一条车道，且要换道，且不在出口范围内
            if vehicle.lane_index in [("upstream", "mid", 0), ("mid", "downstream", 0), ("downstream", "end", 0)] and action[index] == np.int64(2)  and (vehicle.position[0] < self.config["export_of_cav"] or vehicle.position[0] > self.config["export_of_cav"] + self.config["export_length"]):
                action[index] = np.int64(1)
        if self.logger is not None:
            self.logger.log_velocity()
            self.logger.log_vehicle_num(("mid", "downstream", 0))
        action = tuple([*action])
        return super().step(action)

    def close(self):
        if self.logger is not None:
            self.logger.log_final()
        return super().close()

    def _reset(self):
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """创建地图，创建好的道路地图会赋值给self.road"""

        net = RoadNetwork()
        road_length = [400, 200, 200]
        road_width = 3.5
        road = Road(network=net, np_random=self.np_random)
        point = ["upstream", "mid", "downstream", "end"]

        # 第一条车道
        lane_type_of_left = [
            LineType.CONTINUOUS,
            LineType.CONTINUOUS,
            LineType.CONTINUOUS,
        ]
        lane_type_of_right = [LineType.STRIPED, LineType.STRIPED, LineType.STRIPED]
        for i in range(3):
            net.add_lane(
                point[i],
                point[i + 1],
                StraightLane(
                    [sum(road_length[:i]), 0],
                    [sum(road_length[: i + 1]), 0],
                    line_types=[lane_type_of_left[i], lane_type_of_right[i]],
                    speed_limit=22.22,
                ),
            )

        # 第二条车道
        lane_type_of_left = [LineType.NONE, LineType.NONE, LineType.NONE]
        lane_type_of_right = [LineType.STRIPED, LineType.STRIPED, LineType.STRIPED]
        for i in range(3):
            net.add_lane(
                point[i],
                point[i + 1],
                StraightLane(
                    [sum(road_length[:i]), road_width],
                    [sum(road_length[: i + 1]), road_width],
                    line_types=[lane_type_of_left[i], lane_type_of_right[i]],
                    speed_limit=22.22,
                ),
            )

        # 第三条车道
        lane_type_of_left = [LineType.NONE, LineType.NONE, LineType.NONE]
        lane_type_of_right = [LineType.CONTINUOUS, LineType.NONE, LineType.CONTINUOUS]
        for i in range(3):
            net.add_lane(
                point[i],
                point[i + 1],
                StraightLane(
                    [sum(road_length[:i]), 2 * road_width],
                    [
                        sum(road_length[: i + 1]),
                        2 * road_width,
                    ],
                    line_types=[lane_type_of_left[i], lane_type_of_right[i]],
                    speed_limit=22.22,
                ),
            )

        # 岔道
        net.add_lane(
            "mid",
            "downstream",
            StraightLane(
                [sum(road_length[:1]), 3 * road_width],
                [sum(road_length[:2]), 3 * road_width],
                line_types=[LineType.STRIPED, LineType.CONTINUOUS],
                speed_limit=11.11,
            ),
        )
        net.add_lane(
            "downstream",
            "incline",
            StraightLane(
                [sum(road_length[:2]), 3 * road_width],
                [sum(road_length[:2]) + 100, 4 * road_width],
                line_types=[LineType.CONTINUOUS, LineType.CONTINUOUS],
                speed_limit=11.11,
            ),
        )
        net.add_lane(
            "incline",
            "end",
            StraightLane(
                [sum(road_length[:2]) + 100, 4 * road_width],
                [sum(road_length), 4 * road_width],
                line_types=[LineType.CONTINUOUS, LineType.CONTINUOUS],
                speed_limit=11.11,
            ),
        )

        self.road = road

    def _make_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_per_controlled = utils.near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )
        available_lane = [
            ("upstream", "mid", 1),
            ("upstream", "mid", 2),
            ("mid", "downstream", 1),
            ("mid", "downstream", 2),
            ("downstream", "end", 1),
            ("downstream", "end", 2),
            ("downstream", "incline", 0),
            ("incline", "end", 0),
        ]
        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_from="upstream",
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = CustomControlledVehicle(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            count = 0
            while count < others:
                lane = random.choice(available_lane)
                vehicle = RandomVehicle.create_random(
                    self.road,
                    lane_from=lane[0],
                    lane_to=lane[1],
                    lane_id=lane[2],
                    spacing=1 / self.config["vehicles_density"],
                )
                vehicle.randomize_behavior()
                if vehicle.lane_index in available_lane:
                    self.road.vehicles.append(vehicle)
                    count += 1

    def _reward(self, action):
        rewards = []
        for i, vehicle in enumerate(self.controlled_vehicles):
            rewards.append(
                single_reward(
                    vehicle,
                    action[i],
                    self.config["speed_reward_w"],
                    self.config["acceleration_cost_w"],
                    self.config["crash_cost_w"],
                    self.config["lane_change_cost_w"],
                    self.config["export_reward_w"],
                )
            )

        return rewards

    def _is_terminated(self):
        done = [False] * len(self.controlled_vehicles)
        for i, vehicle in enumerate(self.controlled_vehicles):
            if (vehicle.crashed or (self.config["offroad_terminal"] and not vehicle.on_road)):
                done[i] = True
        return done

    def _is_truncated(self):
        if not False in self._is_terminated() or self.time >= self.config["duration"]:
            return True
        else:
            return False


def single_reward(ego: CustomControlledVehicle, action, w1: float, w2: float, w3: float, w4: float, w5: float):
    assert w2 <= 0 and w3 <= 0 and w4 <= 0
    return w1 * speed_reward(ego) + w2 * acceleration_cost(ego) + w3 * crash_cost(ego) + w4 * lane_change_cost(action) + w5 * export_reward(ego)


def speed_reward(ego: CustomControlledVehicle) -> float:
    return ego.speed


def acceleration_cost(ego: CustomControlledVehicle) -> float:
    # NOTE: 由于没有引入时间间隔这个概念，所以使用速度差代替加速度
    # 返回值是正值，所以要取负的系数
    return abs(ego.speed_delta)


def crash_cost(ego: CustomControlledVehicle) -> float:
    if ego.crashed:
        return 100
    else:
        return 0

def lane_change_cost(action) -> float:
    if action in [np.int64(0), np.int64(2)]:
        return 1
    else:
        return 0

def export_reward(ego: CustomControlledVehicle) -> float:
    #if not ego.on_road and ego.lane_index == ("incline", "end", 0):
    #    return 100
    #else:
    #    return 0
    return ego.position[1]