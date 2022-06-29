import gym
import numpy as np
import cityflow

class TSCEnv(gym.Env):
    """
    Environment for Traffic Signal Control task.

    Parameters
    ----------
    world: World object
    agents: list of agent, corresponding to each intersection in world.intersections
    metric: Metric object, used to calculate evaluation metric
    """
    def __init__(self, world, agents, metric):
        self.world = world
        
        self.eng = self.world.eng
        self.n_agents = len(self.world.intersection_ids)
        self.n = self.n_agents

        assert len(agents) == self.n_agents

        self.agents = agents
        action_dims = [agent.action_space.n for agent in agents]
        self.action_space = gym.spaces.MultiDiscrete(action_dims)

        self.metric = metric

    def step(self, actions):
        assert len(actions) == self.n_agents
        r1 = []
        r2 = []

        self.world.step(actions)

        obs = [agent.get_ob() for agent in self.agents]
        rewards = [agent.get_reward() for agent in self.agents]
        for r in rewards:
            r1.append(r[0])
            r2.append(r[1])

        print("\nr1 = " + str(r1) + "\n r2 = " + str(r2) + "\n")

        dones = [False] * self.n_agents
        infos = {"metric": self.metric.update()}
        #infos = {}

        return obs, r1,r2, dones, infos

    def reset(self):
        self.world.reset()
        obs = [agent.get_ob() for agent in self.agents]
        return obs
