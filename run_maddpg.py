import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent.maddpg_agent import MADDPGAgent
from metric import TravelTimeMetric
import argparse
import tensorflow as tf
import os
import logging
from datetime import datetime 

# parse args
def parse_args():
    parser = argparse.ArgumentParser(description='Run Example')
    # Environment
    parser.add_argument('--map', type=str, default='jinan_3_4', help='path of config file')    
    parser.add_argument('--config_file', type=str, default='./data/jinan_3_4/config.json', help='path of config file')    
    parser.add_argument('--data_dir', type=str, default='./data/', help='path of data dir')
    parser.add_argument('--thread', type=int, default=4, help='number of threads')
    parser.add_argument('--steps', type=int, default=500, help='number of steps')#1000 for jinan, 1500 ny_16_3
    parser.add_argument('--action_interval', type=int, default=5, help='how often agent make decisions')
    parser.add_argument('--episodes', type=int, default=300, help='training episodes')#100
    parser.add_argument('--pretrain_episodes', type=int, default=100, help='pre-training episodes')
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.90, help="discount factor")
    parser.add_argument("--epsilon", type=float, default=0.2, help="exploration rate")
    parser.add_argument("--batch-size", type=int, default=256, help="number of batches to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=256, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="model/maddpg/", help="directory in which model should be saved")
    parser.add_argument("--save-rate", type=int, default=3, help="save model once every time this many episodes are completed")
    parser.add_argument("--log-dir", type=str, default="log/maddqg", help="Logs metric by episode")
    return parser.parse_args()

args = parse_args()
args.config_file = os.path.join(args.data_dir, "{}/config.json".format(args.map))
args.save_dir = args.save_dir + "_" + args.map + "/"
args.log_dir = args.log_dir + "_" + args.map

# Initialize logger
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(args.log_dir, datetime.now().strftime('%Y%m%d-%H%M%S') + ".log"))
fh.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(sh)

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(MADDPGAgent(
        action_space,
        LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
        LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
        args,
        i.id
    ))
ob_space_n = []
action_space_n = []
for agent in agents:
    ob_space_n.append(agent.ob_shape)
    action_space_n.append(agent.action_space)
print(ob_space_n)
print(action_space_n)
for i, agent in enumerate(agents):
    agent.build_model(ob_space_n, action_space_n, i)

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)

def pretrain():
    print('Starting pretrain...')
    for e in range(args.pretrain_episodes):
        obs_n = env.reset()
        episode_step = 0
        step = 0
        while step < (args.steps/4):
            if step % args.action_interval == 0:
                # get action
                action_n = [agent.get_action(obs) for agent, obs in zip(agents, obs_n)]
                action_prob_n = [agent.get_action_prob(obs) for agent, obs in zip(agents, obs_n)]
                # environment step
                for _ in range(args.action_interval):
                    new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                    step += 1
                # print(rew_n[0])

                episode_step += 1
                # collect experience
                for i, agent in enumerate(agents):
                    agent.experience(obs_n[i], action_prob_n[i], rew_n[i], new_obs_n[i], done_n[i])
                obs_n = new_obs_n
        if e%10 == 0:
            print("pretraining, episode: {}/{}".format(e, args.pretrain_episodes))



# train maddpg_agent
def train():
    config = tf.ConfigProto(
        intra_op_parallelism_threads=4,
        allow_soft_placement=True,
        log_device_placement=False
    )
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    with sess:
        # Initialize
        sess.run(tf.variables_initializer(tf.global_variables()))
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        train_step = 0
        best_result = evaluate()
        print("initial result: {}".format(best_result))
        pretrain()
        print("buffer length: {}".format(len(agents[0].replay_buffer)))

        print('Starting iterations...')
        for e in range(args.episodes):
            args.epsilon *= 0.99
            episode_rewards = [0.0]  # sum of rewards for all agents
            agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
            obs_n = env.reset()
            episode_step = 0
            step = 0
            while step < args.steps:
                if step % args.action_interval == 0:
                    # get action
                    action_n = [agent.get_action(obs, exploration=True) for agent, obs in zip(agents, obs_n)]
                    action_prob_n = [agent.get_action_prob(obs) for agent, obs in zip(agents, obs_n)]
                    # environment step
                    for _ in range(args.action_interval):
                        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                        step += 1

                    episode_step += 1
                    # collect experience
                    for i, agent in enumerate(agents):
                        agent.experience(obs_n[i], action_prob_n[i], rew_n[i], new_obs_n[i], done_n[i])
                    obs_n = new_obs_n

                    for i, rew in enumerate(rew_n):
                        episode_rewards[-1] += rew
                        agent_rewards[i][-1] += rew

                    # increment global step counter
                    train_step += 1

                    # update all trainers, if not in display or benchmark mode
                    loss = None
                    for agent in agents:
                        loss = agent.update(agents, train_step)
                        # print(loss)
                        # if loss is not None:
                        #     print(loss[0], loss[1])

            print("episode:{}/{}, total agent episode mean reward:{}".format(e, args.episodes, episode_rewards[0]/episode_step))
            # for i in range(len(agents)):
            #     print("agent:{}, episode mean reward:{}".format(i, agent_rewards[i][-1]/episode_step))
            if e % args.save_rate == 0:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                current_result = evaluate()
                logger.info("current_result, episode:{}/{}, avg_travel_time:{}, ".format(e, args.episodes, current_result))
                if current_result < best_result:
                    best_result = current_result
                    saver.save(sess, os.path.join(args.save_dir, "maddpg_{}.ckpt".format(e)))
                    logger.info("best model saved, episode:{}/{}, avg_travel_time:{}".format(e, args.episodes, current_result))

def evaluate():
    obs_n = env.reset()
    step = 0
    while step < args.steps:
        if step % args.action_interval == 0:
            # get action
            action_n = [agent.get_action(obs) for agent, obs in zip(agents, obs_n)]
            for _ in range(args.action_interval):
                obs_n, rew_n, done_n, info_n = env.step(action_n)
                step += 1
    return env.eng.get_average_travel_time()

def test(model_id=None):
    sess = tf.Session()
    with sess:
        # Initialize
        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint(args.save_dir)
        if model_id is not None:
            saver.restore(sess, "model/maddpg/maddpg_{}.ckpt".format(model_id))
        else:
            saver.restore(sess, model_file)
        obs_n = env.reset()
        for i in range(args.steps):
            if i % args.action_interval == 0:
                # get action
                action_n = [agent.get_action(obs) for agent, obs in zip(agents, obs_n)]
                # environment step
                obs_n, rew_n, done_n, info_n = env.step(action_n)
                done = all(done_n)
                if done:
                    break
        logger.info("Final Travel Time is %.4f" % env.eng.get_average_travel_time())

# simulate
train()
#test()
