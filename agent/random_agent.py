
from . import BaseAgent
from generator import LaneVehicleGenerator

class RandAgent(BaseAgent):
    """
    Agent that samples a random action at each delta,
    lower bound for performance 
    """
    def __init__(self, action_space, I, world):
        super().__init__(action_space)
        self.I = I
        self.world = world
        self.world.subscribe("lane_waiting_count")

        # the minimum duration of time of one phase
        self.t_min = 10

        # some threshold to deal with phase requests
        self.min_green_vehicle = 20
        self.max_red_vehicle = 30

    def get_ob(self):
        return None

    def get_action(self, ob):
        lane_waiting_count = self.world.get_info("lane_waiting_count")

        action = self.I.current_phase
        if self.I.current_phase_time >= self.t_min:
            action =  self.action_space.sample()

        return action

    def get_reward(self):
        return None