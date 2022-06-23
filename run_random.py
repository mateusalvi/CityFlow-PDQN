import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import RandAgent
from metric import TravelTimeMetric
import argparse
import os

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--map', type=str, default='newyork_16_7', help='path of config file')    
parser.add_argument('--data_dir', type=str, default='./data/', help='path of config file')    
parser.add_argument('--config_file', type=str, default='./data/jinan_3_4/config.json', help='path of config file')    
parser.add_argument('--thread', type=int, default=2, help='number of threads')
parser.add_argument('--steps', type=int, default=500, help='number of steps')
parser.add_argument('--delta_t', type=int, default=5, help='how often agent make decisions')
args = parser.parse_args()

args.config_file = os.path.join(args.data_dir, "{}/config.json".format(args.map))

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(RandAgent(action_space, i, world))

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)

# simulate
obs = env.reset()
actions = []
for i in range(args.steps):
    actions = []
    for agent_id, agent in enumerate(agents):
        actions.append(agent.get_action(obs[agent_id]))
    obs, rewards, dones, info = env.step(actions)
    print(world.intersections[0]._current_phase, end=",")
    print(env.eng.get_average_travel_time())
    #print(obs)
    #print(rewards)
    # print(info["metric"])

print("Final Travel Time is %.4f" % env.metric.update(done=True))