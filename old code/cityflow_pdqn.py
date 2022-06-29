import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdqn import Estimator, PDQN, Memory
import matplotlib.pyplot as plt
from pathlib import Path
from pareto_q import compute_hypervolume
import csv



class CityflowPDQN(PDQN):

	def start(self, log=None):
		self.observe._is_empty = True
		return super(CityflowPDQN, self).start(log=log)

	def end(self, log=None, writer=None):		
		
		if log.total_steps >= self.learn_start:
			self.epsilon *= 0.999
			self.epsilon = max(self.epsilon, 0.1)

		if (log.episode + 1) % 100 == 0:

			state = self.env.reset()
			
			self.observe._is_empty = True
			obs = self.observe(state)
			q_front = self.q_front(np.expand_dims(obs, axis=0))
			
			try:
				ref_point = np.array([-5, -5])
				hypervolume = compute_hypervolume(q_front[0], self.env.action_space.n, ref_point)
			except ValueError:
				hypervolume = 0

		writer.write("pareto_loss: {}_log_episode:{}\n".format((self.e_loss / log.episode_step), log.episode))
		writer.write("hypervolume_exceeded: {}_log_episode:{}\n".format(self.hypervolume_exceeded, log.episode))

		writer.write("reward_1: {}\n".format(log.reward))
		#writer.write("reward_2: {}\n".format(log.reward[0]))
		"""writer.write("reward_2: {}\n".format(log.reward[0]))
		TypeError: 'float' object is not subscriptable"""

		
		if (log.episode + 1) % 1000 == 0:
			torch.save(self.estimate_objective.model, 'estimate_objective.pt')
			torch.save(self.estimate_reward.model, 'estimate_reward.pt')
			

class ParetoApproximator(nn.Module):

	def __init__(self, nS, nA, nO, device='cpu'):
		super(ParetoApproximator, self).__init__()

		self.nA = nA
		self.nO = nO
		self.device = device

		fc1_in = nS[0] + nO-1
		self.fc1 = nn.Linear(fc1_in, 20)
		self.fc2 = nn.Linear(nA, nA)
		self.fc3 = nn.Linear(20+nA, 10)
		self.out = nn.Linear(10, 1)

	def forward(self, state, point, action):
				
		state = np.array(state, dtype=np.float64)
		state = torch.from_numpy(state)
		point = torch.from_numpy(point)

		inp = torch.cat((state, point), dim=1)                
			
		oh_action = torch.zeros(action.shape[0], self.nA).type(torch.float32).to(self.device)
		oh_action[torch.arange(action.shape[0], device=self.device), action] = 1.

		inp = inp.float()
		fc1 = self.fc1(inp)
		fc1 = F.relu(fc1)

		fc2 = self.fc2(oh_action)
		fc2 = F.relu(fc2)

		fc3 = torch.cat((fc1, fc2), dim=1)
		fc3 = self.fc3(fc3)
		fc3 = F.relu(fc3)

		out = self.out(fc3)

		return out


class RewardApproximator(nn.Module):

	def __init__(self, nS, nA, nO, pareto_approximator=None, device='cpu'):
		super(RewardApproximator, self).__init__()
		#ns				#numero de parametros recebidos
		self.nA = nA    #action space
		self.nO = nO	#observation
		self.device = device

		fc1_in = nS[0]
		self.fc1 = nn.Linear(fc1_in, 20)
		self.fc2 = nn.Linear(nA, nA)
		self.fc3 = nn.Linear(20 + nA, 10)
		self.out = nn.Linear(10, self.nO)

	def forward(self, state, action):
		
		state = np.array(state, dtype=np.float64)
		state = torch.from_numpy(state)
		inp = state
		
		oh_action = torch.zeros(action.shape[0], self.nA).type(torch.float32).to(self.device)
		oh_action[torch.arange(action.shape[0], device=self.device), action] = 1.
		
		inp = inp.float()

		fc1 = self.fc1(inp)
		fc1 = F.relu(fc1)

		fc2 = self.fc2(oh_action)
		fc2 = F.relu(fc2)

		fc3 = torch.cat((fc1, fc2), dim=1)
		fc3 = self.fc3(fc3)
		fc3 = F.relu(fc3)

		out = self.out(fc3)

		return out


def to_chw(whc):
	return np.moveaxis(whc, [1, 0, 2], [2, 1, 0])


class History(object):

	def __init__(self, size=1):
		self.size = size
		self._is_empty = True
		self._type = 'state'
		# will be set in _convert
		self._state = None

	def __call__(self, state):
		state = state
		#state, timestep = state
		# WHC to CHW
		#state = to_chw(state)
		#converted = self.convert(state)
		return state

	def convert(self, state):
		if self._is_empty:
			# add history dimension
			s = np.expand_dims(state, 0)
			# fill history with current state
			self._state = np.repeat(s, self.size, axis=0)
			self._is_empty = False
		else:
			# shift history
			self._state = np.roll(self._state, -1, axis=0)
			# add state to history
			self._state[-1] = state
		return np.concatenate(self._state, axis=0)


class CityflowEstimator(Estimator):

	def predict(self, *net_args, use_target_network=False):
		net = self.target_model if use_target_network else self.model

		preds = net(*net_args)
		
		return preds



def evaluate_reward_estimator(env, observer, model, normalize, device='cpu'):

	state = env.reset()
	obs = observer(state)
	terminal = False
	step = 0
	while not terminal:
		action = np.random.choice(range(2))

		est_reward = model(torch.from_numpy(np.expand_dims(obs, 0).astype(np.float32)).to(device),
						   None,
						   torch.from_numpy(np.expand_dims(action, 0).astype(np.long)).to(device))
		est_reward = est_reward.detach().cpu().numpy() #* normalize
		state, reward, terminal, _ = env.step(action)
		# reward /= normalize
		obs = observer(state)

		print(reward, est_reward)
		step += 1


def train_reward_estimator(env, observer, model, normalize, device='cpu', episodes=10000):
	import tensorboardX as tb
	import datetime
	writer = tb.SummaryWriter(log_dir='runs/rew/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	for episode in range(episodes):
		state = env.reset()
		obs = observer(state)
		terminal = False
		step = 0
		while not terminal:
			action = np.random.choice(range(2))

			est_reward = model(torch.from_numpy(np.expand_dims(obs, 0).astype(np.float32)).to(device),
							   None,
							   torch.from_numpy(np.expand_dims(action, 0).astype(np.long)).to(device))
			est_reward = est_reward.detach().cpu().numpy() # * normalize
			next_state, reward, terminal, _ = env.step(action)
			# reward /= normalize
			next_obs = observer(next_state)

			l = model.update(torch.from_numpy(np.expand_dims(obs, 0).astype(np.float32)).to(device),
							None,
							torch.from_numpy(np.expand_dims(action, 0).astype(np.long)).to(device),
							torch.from_numpy(np.expand_dims(reward, 0).astype(np.float32)).to(device),
							step=1)
			if step % 10 == 0:
				print(f'episode {episode}, step {step} - loss {l}')
				writer.add_scalar('loss', l, episode*200 + step)

			obs = next_obs
			step += 1

		if (episode + 1) % 100 == 0:
			f = Path(list(writer.all_writers.keys())[0]) / 'checkpoints' / 'reward_est_{}.pt'.format(episode)
			f.parents[0].mkdir(parents=True, exist_ok=True)
			torch.save(model.model, f)
	


if __name__ == '__main__':

	import gym
	import gym_cityflow
	import numpy as np
	from pareto_q import action_selection
	import argparse

	parser = argparse.ArgumentParser(description='pareto dqn')
	parser.add_argument('--cityflow-conf', default='/home/mateus/gym_cityflow/gym_cityflow/envs/1x1_config/config.json', type=str)
	parser.add_argument('--lr-reward', default=1e-3, type=float)
	parser.add_argument('--lr-pareto', default=1e-3, type=float)
	parser.add_argument('--batch-size', default=32, type=int)
	parser.add_argument('--copy-reward', default=2000, type=int)
	parser.add_argument('--copy-pareto', default=2000, type=int)
	parser.add_argument('--mem-size', default=250000, type=int)
	parser.add_argument('--learn-start', default=1000, type=int)
	parser.add_argument('--scale', default=1, type=int)
	parser.add_argument('--normalize', action='store_true')
	parser.add_argument('--epsilon-decrease', default=0.999, type=float)
	parser.add_argument('--history-size', default=4, type=int)
	parser.add_argument('--rew-estimator', type=str)
	parser.add_argument('--par-estimator', type=str)
	parser.set_defaults(normalize=False)
	args = parser.parse_args()
	print(args)

	device = 'cpu'

	env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')

	nO = 2
	scales = {1: (87, 66, 3*args.history_size), 2: (174, 132, 3*args.history_size)}
	scales = {1: (20, 18, 1 * args.history_size)}
	nS = (8,) #Alterado de (6,) para (8,) pois estava dando erro na linha 42-43 do pdqb.py

	if args.par_estimator is not None:
		par_model = torch.load(args.par_estimator)
	else:
		par_model = ParetoApproximator(nS, env.action_space.n, nO, device=device).to(device)
	if args.rew_estimator is not None:
		rew_model = torch.load(args.rew_estimator)
	else:
		rew_model = RewardApproximator(nS, env.action_space.n, nO, None, device=device).to(device)

	par_est = CityflowEstimator(par_model, lr=args.lr_pareto, copy_every=args.copy_pareto, clamp=1., device=device)
	rew_est = CityflowEstimator(rew_model, lr=args.lr_reward, copy_every=args.copy_reward, device=device)
	#rew_est = CityflowEstimator(env)
	#rew_est = CityflowReward(env)

	memory = Memory(nS, size=args.mem_size, observation_type='object', nO=nO)
	ref_point = np.array([-10, -10])
	normalize = {'min': np.array([120., -40.]), 'scale': np.array([400.-120, 200.-40])} if args.normalize else None
	epsilon_decrease = args.epsilon_decrease

	agent = CityflowPDQN(env, policy=lambda s, q, e: action_selection(s, q, e, ref_point),  # np.random.choice(np.arange(4)),
					 memory=memory,
					 observe=History(size=args.history_size),
					 estimate_reward=rew_est,
					 estimate_objective=par_est,
					 normalize_reward=normalize,
					 nO=nO,
					 learn_start=args.learn_start,
					 batch_size=32,
					 gamma=1.)

	logdir = 'runs/cityflow/'
	
	agent.train(5, logdir=logdir)
	


	