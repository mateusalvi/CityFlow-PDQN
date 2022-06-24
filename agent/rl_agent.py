from . import BaseAgent

class RLAgent(BaseAgent):
    def __init__(self, action_space, ob_generator, reward_generator,reward_generator2):
        super().__init__(action_space)
        self.ob_generator = ob_generator
        self.reward_generator = reward_generator
        self.reward_generator2 = reward_generator2

    def get_ob(self):
        return self.ob_generator.generate()

    def get_reward(self):
        reward = self.reward_generator.generate()
        print("\n\n\n\nreward\n\n\n\n")
        print(reward)
        print("\n\n\n\n\n\n\n\n")

        
        reward2 = self.reward_generator2.generate()

        print("\n\n\n\nreward2\n\n\n\n")
        print(reward2)
        print("\n\n\n\n\n\n\n\n")
        
        assert len(reward) == 1
        assert len(reward2) == 1
        return reward[0],reward2[0]

    def get_action(self, ob):
        return self.action_space.sample()