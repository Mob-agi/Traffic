from typing import Tuple, Union

import numpy as np
from highway_env import utils
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle


class RandomVehicle(IDMVehicle):
    def __init__(self, road, position, heading=0, speed=0, target_lane_index=None, target_speed=None, route=None,
               enable_lane_change=True, timer=None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route, enable_lane_change,
                       timer)

    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """
        self.speed = self.lane.speed_limit
        if self.target_lane_index in [("upstream", "mid", 0), ("mid", "downstream", 0), ("downstream", "end", 0)]:
            self.target_lane_index = self.lane_index
            return
        super().change_lane_policy()


class CustomControlledVehicle(MDPVehicle):

    def __init__(self, road, position, heading=0, speed=0, target_lane_index=None, target_speed=None, target_speeds=None,
               route=None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, target_speeds, route)
        self.t = 0
        self.last_lane = self.lane_index
        self.lane_list = np.zeros(50)
        self.speed_delta: float

    def act(self, action: Union[dict, str] = None) -> None:
        speed = self.speed
        super().act(action)
        self.speed_delta = self.speed - speed
        return

    def on_state_update(self):
        super().on_state_update()
        if self.speed > self.lane.speed_limit:
            self.speed = self.lane.speed_limit