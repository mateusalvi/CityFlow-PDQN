from . import RLAgent
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop,SGD

#from keras.optimizers import RMSprop, SGD
import os

class PDQNAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, reward_generator2, iid):
        super().__init__(action_space, ob_generator, reward_generator, reward_generator2)

        self.iid = iid

        self.ob_length = ob_generator.ob_length

        self.memory = deque(maxlen=2000)
        self.learning_start = 2000
        self.update_model_freq = 1
        self.update_target_model_freq = 20

        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.batch_size = 32

        

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

#########################  REUTILIZAR?  ####################################
    def compute_hypervolume(q_set, nA, ref):
        q_values = np.zeros(nA)
        for i in range(nA):
            # pygmo uses hv minimization,
            # negate rewards to get costs
            points = np.array(q_set[i]) * -1.
            hv = hypervolume(points)
            # use negative ref-point for minimization
            q_values[i] = hv.compute(ref*-1)
        return q_values
#############################################################################

    #########################  ALTERAR CONFORME O PAPER  ####################################
    def get_action(self, q_set,ob):
        
        hv = compute_hypervolume(q_set, q_set.shape[0], ref = [-10,-10])
        
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        #ob = self._reshape_ob(ob)
        
        
       # act_values = self.model.predict(ob)

        action = self.epsilon - hv

        return action[0]

        
     #############################################################################   
        

        

        """
        SOFTMAX
        def action_selection(state, q_set, epsilon, ref):
            q_values = compute_hypervolume(q_set, q_set.shape[0], ref)

            denominator = 0
            probability_vector = []

            for q_value in q_values: #calculate the denominator for softmax (sum of e^q_value)
                denominator += (np.e ** (q_value/epsilon))
            for q_value in q_values: #calculate the choice probability for each q_value to an array of probabilities
                probability_vector.append(((np.e ** (q_value/epsilon))/(denominator)))

            action = np.argwhere(q_values == (np.random.choice(q_values, 1, probability_vector))).flatten() #select an action (q_values) biased by its probability (probability_vector)

            return action[0]
        """


    def sample(self):
        return self.action_space.sample()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(20, input_dim=self.ob_length, activation='relu'))
        #model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_space.n, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=RMSprop()
        )
        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, ob, action, reward, next_ob):
        self.memory.append((ob, action, reward, next_ob))

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        obs, actions, rewards, next_obs = [np.stack(x) for x in np.array(minibatch).T]
        #Qset equação 3 do paper
        #Qset = VetorDeRewards +(elemento a elemento) gama*NonDominated
        #Qset = np.add(rewards + self.gamma*self.target_model.predict(next_obs))
        #Problema no shape size (32,) + (32,8)
        target = rewards + self.gamma * np.amax(self.target_model.predict(next_obs), axis=1)
        
        
        target_f = self.model.predict(obs)
        #O Qset tem que passar pro hypervolume
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
            #target_f[i][action] = Qset[i]
        history = self.model.fit(obs, target_f, epochs=1, verbose=0)
        
        #print(history.history['loss'])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/dqn"):
        name = "dqn_agent_{}.h5".format(self.iid)
        model_name = os.path.join(dir, name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/dqn"):
        name = "dqn_agent_{}.h5".format(self.iid)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)