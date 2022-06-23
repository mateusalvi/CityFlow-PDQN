import sys
import torch
import matplotlib.pyplot as plt
from pareto_q import Agent, compute_hypervolume, action_selection
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import copy
from metric import TravelTimeMetric
from pathlib import Path
plt.switch_backend('agg')

Transition = namedtuple('Transition',
                        ['observation',
                         'action',
                         'reward',
                         'next_observation',
                         'terminal'])

class Memory(object):

    def __init__(self, observation_shape, observation_type='float16', size=1000000, nO=1):
        self.current = 0
        # we will only save next_states,
        # as current state is simply the previous next state.
        # We thus need an extra slot to prevent overlap between the first and
        # last sample
        size += 1
        self.size = size

        self.actions = np.empty((size,), dtype='uint8')
        if observation_shape == (1,):
            self.next_observations = np.empty((size,), dtype=observation_type)
        else:
            self.next_observations = np.empty((size,) + observation_shape, dtype=observation_type)
        self.rewards = np.empty((size, nO), dtype='float16')
        self.terminals = np.empty((size,), dtype=bool)

    def add(self, transition):
        # first sample, need to save current state
        if self.current == 0:
            self.next_observations[0] = transition.observation

        self.current += 1
        current = self.current % self.size
        self.actions[current] = transition.action
        self.next_observations[current] = transition.next_observation
        self.rewards[current] = transition.reward
        self.terminals[current] = transition.terminal

    def sample(self, batch_size):
        assert self.current > 0, 'need at least one sample in memory'
        high = self.current % self.size
        # did not fill memory
        if self.current < self.size:
            # start at 1, as 0 contains only current state
            low = 1
        else:
            # do not include oldest sample, as it's state (situated in previous sample)
            # has been overwritten by newest sample
            low = high - self.size + 2
        indexes = np.empty((batch_size,), dtype='int32')
        i = 0
        while i < batch_size:
            # include high
            s = np.random.randint(low, high+1)
            # cannot include first step of episode, as it does not have a previous state
            if not self.terminals[s-1]:
                indexes[i] = s
                i += 1
        batch = Transition(
            self.next_observations[indexes-1],
            self.actions[indexes],
            self.rewards[indexes],
            self.next_observations[indexes],
            self.terminals[indexes]
        )

        return batch


class Estimator(object):

    def __init__(self, model, lr=1e-3, tau=1., copy_every=0, clamp=None, device='cpu'):
        self.model = model #rede neural
        self.target_model = copy.deepcopy(model) #same
        self.device = device
        self.copy_every = copy_every
        self.tau = tau
        self.clamp = clamp
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0)
        self.loss = nn.MSELoss(reduction='none')

    def should_copy(self, step):
        return self.copy_every and not step % self.copy_every

    def update_target(self, tau):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

    def predict(self, *net_args, use_target_network=False):
        net = self.target_model if use_target_network else self.model
        preds = net(*net_args)
        return preds

    def __call__(self, *net_args, use_target_network=False):
        return self.predict(*net_args, use_target_network=use_target_network).detach().cpu().numpy()

    def update(self, targets, *net_args, step=None):
        self.opt.zero_grad()    
        
        preds = self.predict(*net_args, use_target_network=False)
                       
        
        l = self.loss(preds, torch.from_numpy(targets).to(self.device))
        
        if self.clamp is not None:
            l = torch.clamp(l, min=-self.clamp, max=self.clamp)
        l = l.mean()

        l.backward()

        if step % 100 == 0:
            total_norm = 0
            for p in self.model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.total_norm = total_norm
        
        self.opt.step()

        if self.should_copy(step):
            self.update_target(self.tau)

        return l


class PDQN(Agent):

    def __init__(self, env,
                 policy=None,
                 memory=None,   
                 observe=lambda x: x,
                 estimate_reward=None,
                 estimate_objective=None,
                 normalize_reward=None,
                 epsilon=1., ############################################## PLACEHOLDER: ALSO USED AS TEMPERATURE FOR SOFTMAX
                 batch_size=1,
                 learn_start=0,
                 nO=2,
                 gamma=1.,
                 n_samples=10):
        self.env = env
        self.observe = observe
        self.policy = policy
        self.estimate_objective = estimate_objective
        self.estimate_reward = estimate_reward
        self.memory = memory        
        self.nO = nO
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.learn_start = learn_start
        self.n_samples = n_samples

        if normalize_reward is None:
            normalize_reward = {'min': np.ones(nO), 'scale': np.ones(nO)}
        self.normalize_reward = normalize_reward

        self.hypervolume_exceeded = 0

    def start(self, log=None):

        self.e_loss = 0
        state = self.env.reset()
        obs = self.observe(state)


        return {'observation': obs,
                'terminal': False}

    def sample_points(self, n=10):
        o_samples = []
        for o in range(self.nO-1):
            # sample assuming normalized scale, add noise # 24 / self.normalize_reward[0]
            o_sample = np.linspace(0, 1, n) + np.random.normal(0, 0.01, size=n)
            o_samples.append(np.expand_dims(o_sample, 1))
        return np.concatenate(o_samples, axis=1)

    def pareto_front(self, obs, samples, use_target_network=False):


        samples = samples.astype(np.float32)
        n_samples = len(samples)
        batch_size = len(obs)
        obs_dims = len(obs.shape) - 1

        
        obs = np.tile(obs, (n_samples,) + (1,)*obs_dims) #duplicate to number of samples (32 * samples, 6)
        
        samples = np.repeat(samples, batch_size, axis=0) #duplica xbatchsize


        a_obs = np.tile(obs, (self.env.action_space.n,) + (1,)*obs_dims) #obs * n_actions
        a_samples = np.tile(samples, (self.env.action_space.n, 1)) #samples * n_actions
        as_ = np.repeat(np.arange(self.env.action_space.n), n_samples*batch_size) #actions for each sample

        oa_obj = self.estimate_objective(a_obs,
                                         a_samples,
                                         as_,
                                         use_target_network=use_target_network)
        # add predicted objective to samples

        oa_front = np.concatenate((a_samples, oa_obj), axis=1)
        
        
        # [nA nSamples Batch nO]
        oa_front = oa_front.reshape(self.env.action_space.n, n_samples, batch_size, -1)
      
        # [Batch nA nSamples nO]
        oa_front = np.moveaxis(oa_front, [0, 1, 2], [1, 2, 0])
        return oa_front

    def q_front(self, obs, n=20, use_target_network=False):

        samples = self.sample_points(n)

        front = self.pareto_front(obs, samples, use_target_network=use_target_network)

        obs_dims = len(obs.shape) - 1

        oa = np.tile(obs, (self.env.action_space.n,) + (1,)*obs_dims)
        as_ = np.repeat(np.arange(self.env.action_space.n), len(obs))
        # shape [nA*Batch nO]
        

        r_pred =  self.estimate_reward(oa,
                                     as_.astype(np.long),
                                     use_target_network=use_target_network)

        
        # normalize reward for pareto_estimator
        r_pred = (r_pred - self.normalize_reward['min']) / self.normalize_reward['scale']

        # reshape to [Batch nA 1 nO] so it can be added to front (with shape [Batch nA nSamples nO])
        r_pred = np.moveaxis(r_pred.reshape(self.env.action_space.n, len(obs), 1, -1), [0, 1], [1, 0])

        # q_pred = r_pred + self.gamma*front
        # TEST try to keep the same range for obj 0, shift values accordingly

        q_pred = self.gamma * front
        q_pred[:, :, :, -1] += r_pred[:, :, :, -1]

        # shifts = (r_pred[:, :, 0, 0]/(124/self.normalize_reward[0]/n)).astype(int)
        shifts = np.abs(q_pred[:, :, :, 0] - r_pred[:, :, :, 0]).argmin(axis=2)
        for b_i in range(r_pred.shape[0]):
            for a_i in range(r_pred.shape[1]):
                if shifts[b_i, a_i]:
                    # shift the values according to corresponding shift, fill leftmost values with first shifted element
                    q_pred[b_i, a_i, shifts[b_i, a_i]:] = q_pred[b_i, a_i, :-shifts[b_i, a_i]]
                    q_pred[b_i, a_i, :shifts[b_i, a_i], -1] = q_pred[b_i, a_i, shifts[b_i, a_i], -1]
                    # add immediate reward to shifted elements
                    q_pred[b_i, a_i, shifts[b_i, a_i]:, 0] += r_pred[b_i, a_i, 0, 0]
        # pessimistic bias, assume normalized reward # -20/self.normalize_reward[1]    

        q_pred += np.array([0, -1]).reshape(1, 1, 1, 2)
        return q_pred

    def step(self, previous, log=None):

        if log.total_steps >= 2*self.learn_start:
            # one-sized batch
            
            q_front = self.q_front(np.expand_dims(np.array(previous['observation']),0), n=self.n_samples, use_target_network=False)
            #q_front = self.q_front(np.array(previous['observation']), n=self.n_samples, use_target_network=False)
            
            try:
                if log.total_steps % 3 == 0:
                    action = self.policy(previous['observation'], q_front[0], self.epsilon)
                    self.action_on = action
                else:
                    action = self.action_on


            except ValueError:
                if log.total_steps % 3 == 0:
                    action = np.random.choice(range(self.env.action_space.n))
                    self.hypervolume_exceeded += 1
                    self.action_on = action
                else:
                    action = self.action_on
        else:

            if log.total_steps % 3 == 0:
                action = np.random.choice(range(self.env.action_space.n))
                self.action_on = action
            elif log.total_steps == 0:
                action = np.random.choice(range(self.env.action_space.n))
                self.action_on = action
            else:
                action = self.action_on
                        
        next_state, reward, terminal, _ = self.env.step(action)
        next_obs = self.observe(next_state)

        # add in replay memory
        t = Transition(observation=previous['observation'],
                       action=action,
                       reward=reward,
                       next_observation=next_obs,
                       terminal=terminal)

        self.memory.add(t)

        if log.total_steps >= self.batch_size: # self.learn_start:


            batch = self.memory.sample(self.batch_size)    

            # normalize reward for pareto_estimator

            batch_rew_norm = (batch.reward - self.normalize_reward['min'])/self.normalize_reward['scale']

            
            batch_q_front_next = self.q_front(batch.next_observation, n=self.n_samples, use_target_network=True)

            # convert to float
            batch_observation = batch.observation # .astype(np.float32)
            batch_q_front_next = batch_q_front_next.astype(np.float32)



            # as samples are more or less on same coords for each action, take the best ones across actions
            batch_q_front_next = np.moveaxis(batch_q_front_next, [0, 1, 2], [0, 2, 1])



            # batch_q_front_next[:, :, 1:3, 1] to restrict to actions 1, 2 (right, down)
            b_max = np.argmax(batch_q_front_next[:, :, :, 1], axis=2)


            b_indices, s_indices = np.indices(batch_q_front_next.shape[:2])
            # b_max + 1 if using actions 1, 2
            batch_q_front_next = batch_q_front_next[b_indices, s_indices, b_max]
           
            # non-dominated across all actions

            batch_non_dominated = []
            batch_observations = []
            batch_actions = []
            # add pareto front for each sample
            
            for i, b_i in enumerate(batch_q_front_next):

                # TODO TEST before learn start, initialize net pessimistically
                if log.total_steps < self.learn_start:
                    non_dominated = b_i
                    non_dominated[:, -1] = -1.
                    
                    
                # if state-action leads to terminal next_state, next_state has no pareto front, only immediate reward
                elif batch.terminal[i]:
                    
                    min_ = np.abs(b_i[:, 0] - batch_rew_norm[i][0]).argmin()
                    non_dominated = b_i
                    non_dominated[:min_, -1] = batch_rew_norm[i][-1]
                    non_dominated[min_] = batch_rew_norm[i]
                    non_dominated[min_+1:, -1] = -1.

                    # non_dominated = batch.reward[i].reshape(1, -1)
                else:
                    
                    non_dominated = b_i
                    
                    # non_dominated = get_non_dominated(b_i) # + batch.reward[i].reshape(1, -1)
                    # TODO why all the time (<- indent left)
                    
                    non_dominated += np.array([0, 1]).reshape(1, 2)

                #########
                batch_non_dominated.append(non_dominated)
                
                # batch_non_dominated = np.concatenate((batch_non_dominated, non_dominated))
                observation = np.tile(batch_observation[i], (len(non_dominated), *([1]*len(batch_observation[0].shape))))
                batch_observations.append(observation)
                # batch_observations = np.concatenate((batch_observations, observation))
                action = np.repeat(batch.action[i], len(non_dominated))
                batch_actions.append(action)
                # batch_actions = np.concatenate((batch_actions, action))
            
            batch_non_dominated = np.concatenate(batch_non_dominated)  
            batch_observations = np.concatenate(batch_observations)
            batch_actions = np.concatenate(batch_actions)

            # add n-1 dims of points as input dimensions, net will need to predict the last dim
            # assume observations are 1D
            # batch_observations = np.concatenate((batch_observations, batch_non_dominated[:, :-1]), axis=1)
            

            e_loss = self.estimate_objective.update(batch_non_dominated[:, -1:],
                                                    batch_observations,
                                                    batch_non_dominated[:, :-1],
                                                    batch_actions.astype(np.long),
                                                    step=log.total_steps)

            self.e_loss += e_loss
            # unnormalized reward for reward_estimator
            self.estimate_reward.update(batch.reward.astype(np.float32),
                                        batch_observation,
                                        batch.action.astype(np.compat.long),
                                        step=log.total_steps)

            if log.total_steps % 100 == 0:

                self.writer.write("grad_norm= {} total_steps= {}\n".format(self.estimate_objective, log.total_steps))
                #self.writer.add_scalar('grad_norm', self.estimate_objective.total_norm, log.total_steps)

            if log.total_steps % 199 == 0:
                # unnormalized reward for reward_estimator
                
                r_pred = self.estimate_reward(batch_observation,
                                              batch.action.astype(np.compat.long),
                                              use_target_network=False)
            
                for oi in range(self.nO):
                    r_diff = np.mean((r_pred[:, oi] - batch.reward[:, oi])**2)
                    self.writer.write("reward_{}_predict = {} | log_total_steps={}\n".format(oi, np.mean(r_pred[:, oi]), log.total_steps))
                    
                    self.writer.write("reward_{}_predict_diff={} | log_total_steps={}\n".format(oi, r_diff, log.total_steps))
                    

        return {'observation': next_obs,
                'reward': reward,
                'terminal': terminal}

    def end(self, log=None, writer=None):
        
        if log.total_steps >= self.learn_start:
            self.epsilon *= epsilon_decrease
            self.epsilon = max(self.epsilon, 0.1)
        if (log.episode + 1) % 100 == 0:

            # all states of deep-sea-treasure
            plot_states = list(range(10)) + \
                          list(range(11, 20))+ \
                          list(range(22, 30))+ \
                          list(range(33, 40)) + \
                          list(range(46, 50))+ \
                          list(range(56, 60))+ \
                          list(range(66, 70))+ \
                          list(range(78, 80))+ \
                          list(range(88, 90))+ \
                          list(range(99, 100))
            # estimate pareto front for all states
            obs = np.array([]).reshape(0, self.env.nS)
            for s in plot_states:
                obs = np.concatenate((obs, np.expand_dims(self.observe(s), 0)))
            q_fronts = self.q_front(obs, self.n_samples)
            # undo pessimistic bias
            q_fronts += np.array([0, 1]).reshape(1, 1, 1, 2)
            # unnormalize reward
            q_fronts = q_fronts * self.normalize_reward['scale'].reshape(1, 1, 1, 2) + self.normalize_reward['min'].reshape(1, 1, 1, 2)

            try:
                ref_point = np.array([-2, -2])
                hypervolume = compute_hypervolume(q_fronts[0], self.env.nA, ref_point)
            except ValueError:
                hypervolume = 0

            act = 2
            fig, axes = plt.subplots(11, 10, sharex='col', sharey='row',
                                     subplot_kw={'xlim': [-1, 150], 'ylim': [-20, 1]})

            fig.subplots_adjust(wspace=0, hspace=0)
            for s in range(len(plot_states)):
                ax = axes[np.unravel_index(plot_states[s], (11, 10))]
                x = q_fronts[s, act, :, 0]
                y = q_fronts[s, act, :, 1]
                ax.plot(x, y)

                # true pareto front
                true_xy = true_non_dominated[plot_states[s]][act]
                true_xy = true_xy * self.normalize_reward['scale'].reshape(1, 2) + self.normalize_reward['min'].reshape(1, 2)

                ax.plot(true_xy[:, 0], true_xy[:, 1], '+')
            for s in range(self.env.nS):
                if unreachable(s):
                    ax = axes[np.unravel_index(s, (11, 10))]
                    ax.set_facecolor((1.0, 0.47, 0.42))

            #plt.show()
            fig.canvas.draw()
            # Now we can save it to a numpy array.
            #data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            #data = np.rollaxis(data, -1, 0)
            plt.savefig("pdqn.png")
            #writer.add_scalar('hypervolume', np.amax(hypervolume), log.total_steps)
            #writer.add_image('pareto_front', data, log.total_steps)
            #writer.add_scalar('epsilon', self.epsilon, log.total_steps)
            plt.close(fig)
        #writer.add_scalar('pareto_loss', self.e_loss / log.episode_step, log.episode)
        #writer.add_scalar('hypervolume_exceeded', self.hypervolume_exceeded, log.episode)

        if (log.episode + 1) % 1000 == 0:
            f = Path(list(writer.all_writers.keys())[0]) / 'checkpoints' / 'reward_est_{}.pt'.format(log.episode)
            f.parents[0].mkdir(parents=True, exist_ok=True)
            torch.save(self.estimate_reward.model, f)

            f = Path(list(writer.all_writers.keys())[0]) / 'checkpoints' / 'pareto_est_{}.pt'.format(log.episode)
            f.parents[0].mkdir(parents=True, exist_ok=True)
            torch.save(self.estimate_objective.model, f)


def get_non_dominated(solutions):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1

    return solutions[is_efficient]


def one_hot(env, s):
    oh = np.zeros(env.nS, dtype=np.float32)
    oh[s] = 1
    return oh


class ParetoApproximator(nn.Module):

    def __init__(self, nS, nA, nO, device='cpu'):
        super(ParetoApproximator, self).__init__()

        self.nA = nA
        self.nO = nO
        self.device = device

        fc1_in = nS + nO-1 # 3*conv1_h*conv1_w
        self.fc1 = nn.Linear(fc1_in, fc1_in // 2)
        # self.fc1b = nn.Linear(fc1_in, fc1_in)
        self.fc2 = nn.Linear(nA, nA)
        self.fc3 = nn.Linear(fc1_in // 2+nA, fc1_in // 2)
        # self.fc3b = nn.Linear(fc1_in, fc1_in)
        self.out = nn.Linear(fc1_in // 2, 1)

    def forward(self, state, point, action):
        # conv1 = self.conv1(state)
        # conv1 = F.relu(conv1)
        # b, c, h, w = conv1.shape
        # fc1 = self.fc1(conv1.view(b, c*w*h))
        inp = torch.cat((state.float(), point), dim=1)
        # inp = state_point
        oh_action = torch.zeros(action.shape[0], self.nA).type(torch.float32).to(self.device)
        oh_action[torch.arange(action.shape[0], device=self.device), action] = 1.

        fc1 = self.fc1(inp)
        fc1 = F.relu(fc1)
        #fc1 = self.fc1b(fc1)
        #fc1 = F.relu(fc1)

        fc2 = self.fc2(oh_action)
        fc2 = F.relu(fc2)

        fc3 = torch.cat((fc1, fc2), dim=1)
        fc3 = self.fc3(fc3)
        fc3 = F.relu(fc3)
        #fc3 = self.fc3b(fc3)
        #fc3 = F.relu(fc3)

        out = self.out(fc3)
        return out


class RewardApproximator(nn.Module):

    def __init__(self, nS, nA, nO, device='cpu'):
        super(RewardApproximator, self).__init__()

        self.nA = nA
        self.nO = nO
        self.device = device

        fc1_in = int(nS)
        self.fc1 = nn.Linear(fc1_in, fc1_in // 2)
        # self.fc1b = nn.Linear(fc1_in, fc1_in)
        self.fc2 = nn.Linear(nA, nA)
        self.fc3 = nn.Linear(fc1_in //2+nA, fc1_in // 2)
        # self.fc3b = nn.Linear(fc1_in, fc1_in)
        self.out = nn.Linear(fc1_in // 2, self.nO)

    def forward(self, state, action):
        inp = state.float()
        # inp = state_point
        oh_action = torch.zeros(action.shape[0], self.nA).type(torch.float32).to(self.device)
        oh_action[torch.arange(action.shape[0], device=self.device), action] = 1.

        fc1 = self.fc1(inp)
        fc1 = F.relu(fc1)
        #fc1 = self.fc1b(fc1)
        #fc1 = F.relu(fc1)

        fc2 = self.fc2(oh_action)
        fc2 = F.relu(fc2)

        fc3 = torch.cat((fc1, fc2), dim=1)
        fc3 = self.fc3(fc3)
        fc3 = F.relu(fc3)
        #fc3 = self.fc3b(fc3)
        #fc3 = F.relu(fc3)

        out = self.out(fc3)
        return out


class DSTReward(object):

    def __init__(self, env):
        self.env = env
        self.model = None

    def __call__(self, *args, use_target_network=False):
        oa, as_ = args
        r_pred = np.zeros((len(oa), 2))
        # override predictions with actual immediate reward
        for i in range(len(oa)):
            state = np.argwhere(oa[i] == 1)[0, 0]
            state_coord = np.unravel_index(state, self.env.shape)
            if as_[i] == 2 and (state_coord[0] + 1, state_coord[1]) in self.env._treasures():
                r_pred[i, 0] = self.env._treasures()[(state_coord[0] + 1, state_coord[1])]  # /self.normalize_reward[0]
            else:
                r_pred[i, 0] = 0
            r_pred[i, 1] = -1  # /self.normalize_reward[1]

        return r_pred

    def update(self, targets, *net_args, step=None):
        return np.zeros(len(targets))


def evaluate_agent(agent, env, save_dir, true_non_dominated=None):
    f = Path(save_dir) / 'evaluation'
    print(str(f))
    f.mkdir(parents=True, exist_ok=True)

    def policy(a, obs, q_front, target):
        d = np.linalg.norm(target-q_front, axis=2)
        i = np.unravel_index(np.argmin(d), q_front.shape[:-1])
        p = q_front[i]
        action = i[0]
        r_pred = a.estimate_reward(np.expand_dims(obs, 0), np.array([[action]]), use_target_network=False)
        return action, p-r_pred[0]/agent.normalize_reward

    state = env.reset()
    obs = agent.observe(state)

    q_front = agent.q_front(np.expand_dims(np.array(obs), 0), use_target_network=False)
    initial_pareto = np.concatenate(q_front[0], axis=0)
    initial_pareto = get_non_dominated(initial_pareto)
    print(initial_pareto*agent.normalize_reward)
    pi = int(input('choose point index: '))
    target = initial_target = initial_pareto[pi]
    print(target*agent.normalize_reward)

    terminal = False
    step = 0
    episode_reward = 0
    while not terminal:
        q_front = agent.q_front(np.expand_dims(np.array(obs), 0), use_target_network=False)
        action, new_target = policy(agent, obs, q_front[0], target)
        # action = agent.policy(obs, q_front[0], 0.)
        next_state, reward, terminal, _ = env.step(action)
        next_obs = agent.observe(next_state)

        # plot q_front
        fig, axes = plt.subplots(1, 4, sharex='col', sharey='row', )
        fig.subplots_adjust(wspace=0, hspace=0)
        for act in range(4):
            ax = axes[act]  # np.unravel_index(act, (1, 2))]
            x = q_front[0, act, :, 0] * agent.normalize_reward[0]
            y = q_front[0, act, :, 1] * agent.normalize_reward[1]
            ax.plot(x, y)
        axes[action].scatter([target[0]* agent.normalize_reward[0]], [target[1]* agent.normalize_reward[1]], c='red')
        plt.savefig(f'{str(f)}/step_{step}.png')
        plt.close()

        # print(reward)
        target = new_target
        episode_reward += reward
        obs = next_obs
        step += 1
    print(episode_reward)

    plt.figure()
    initial_pareto = initial_pareto * agent.normalize_reward
    plt.scatter(initial_pareto[:, 0], initial_pareto[:, 1])
    if true_non_dominated is not None:
        true_pareto_front = np.concatenate(true_non_dominated[0], axis=0)
        true_pareto_front = get_non_dominated(true_pareto_front)
        true_pareto_front *= normalize
        plt.scatter(true_pareto_front[:, 0], true_pareto_front[:, 1])
    plt.scatter([initial_target[0]* agent.normalize_reward[0]], [initial_target[1]* agent.normalize_reward[1]], c='red')
    plt.scatter([episode_reward[0]], [episode_reward[1]], c='green')
    plt.savefig(f'{str(f)}/comparison.png')
    plt.close()




if __name__ == '__main__':
    import deep_sea_treasure
    import gym
    import argparse

    parser = argparse.ArgumentParser(description='pareto dqn')
    parser.add_argument('--lr-reward', default=1e-3, type=float)
    parser.add_argument('--lr-pareto', default=3e-4, type=float)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--copy-reward', default=1, type=int)
    parser.add_argument('--copy-pareto', default=100, type=int)
    parser.add_argument('--mem-size', default=250000, type=int)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--epsilon-decrease', default=0.999, type=float)
    parser.add_argument('--rew-estimator', type=str)
    parser.add_argument('--par-estimator', type=str)
    parser.add_argument('--n-samples', type=int, default=10)
    parser.set_defaults(normalize=False)

    args = parser.parse_args()
    print(args)
    device = 'cpu'
    env = gym.make('deep-sea-treasure-v0')
    nO = 2
    if args.par_estimator is not None:
        par_model = torch.load(args.par_estimator)
        par_model.device = 'cpu'
    else:
        par_model = ParetoApproximator(env.nS, env.nA, nO, device=device).to(device)
    if args.rew_estimator is not None:
        rew_model = torch.load(args.rew_estimator)
    else:
        rew_model = RewardApproximator(env.nS, env.nA, nO, device=device).to(device)
    rew_est = Estimator(rew_model, lr=args.lr_reward, copy_every=args.copy_reward)
    # rew_est = DSTReward(env)
    par_est = Estimator(par_model, lr=args.lr_pareto, copy_every=args.copy_pareto)

    memory = Memory((env.nS,), size=args.mem_size, nO=nO)
    ref_point = np.array([-2, -2])
    normalize = {'min': np.array([0,0]), 'scale': np.array([124, 19])} #if args.normalize else None
    epsilon_decrease = args.epsilon_decrease

    true_non_dominated = dst_non_dominated(env, normalize)

    agent = PDQN(env, policy=lambda s, q, e: action_selection(s, q, e, ref_point),
                 memory=memory,
                 observe=lambda s: one_hot(env, s),
                 estimate_reward=rew_est,
                 estimate_objective=par_est,
                 normalize_reward=normalize,
                 nO=nO,
                 learn_start=100,
                 batch_size=1,
                 gamma=1.,
                 n_samples=args.n_samples)

    logdir = 'runs/pareto-dqn/'
    #evaluate_agent(agent, env, logdir, true_non_dominated)
    agent.train(150, logdir=logdir)