import numpy as np
from scipy.special import softmax
from vowpalwabbit import pyvw

# Stationary CB policies
# 1. UniRan
# 2. LeastLoad
# 3. HighestLoad

class UniRan:
    """
    UniRan: uniformly randomly 
    """
    def __init__(self, num_actions, horizon):
        self.policy_type = 'C' 
        self.num_actions = num_actions
        self.horizon = horizon
        self.probability = 1 / num_actions
        self.probabilities = [1 / num_actions] * num_actions

    def update(self, action, reward):
        pass

    def choose_action(self, context, return_prob=False, behavior_policy=True, chosen_action=None):

        if return_prob:
            return self.probabilities

        action = np.random.choice(self.num_actions)

        return action, self.probability

    def reset(self):
        pass

class LeastLoad:
    def __init__(self, num_actions, epsilon=0):
        self.policy_type = 'C'
        self.num_actions = num_actions
        self.epsilon = epsilon

    def update(self, action, reward):
        pass

    def choose_action(self, context, return_prob=False, behavior_policy=True, chosen_action=None):
        least_loaded_server = np.argmin(context)
        probabilities = [self.epsilon / (self.num_actions - 1)] * self.num_actions
        probabilities[least_loaded_server] = 1 - self.epsilon

        if return_prob:
            return probabilities
        action = np.random.choice(self.num_actions, p=probabilities)

        return action, probabilities[action]

    def reset(self):
        pass

class HighestLoad:
    def __init__(self, num_actions, epsilon=0):
        self.policy_type = 'C'
        self.num_actions = num_actions
        self.epsilon = epsilon

    def update(self, action, reward):
        pass

    def choose_action(self, context, return_prob=False, behavior_policy=True, chosen_action=None):
        highest_loaded_server = np.argmax(context)
        probabilities = [self.epsilon / (self.num_actions - 1)] * self.num_actions
        probabilities[highest_loaded_server] = 1 - self.epsilon

        if return_prob:
            return probabilities
        action = np.random.choice(self.num_actions, p=probabilities)

        return action, probabilities[action]

    def reset(self):
        pass

# Nonstationary Multi-Armed Bandit policies
# 1. EpsilonGreedy
# 2. UCB1
# 3. GradBandit
# 4. NonstationaryBandit

class EpsilonGreedy:
    def __init__(self, num_actions, epsilon):
        self.policy_type = 'M'
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.num_actions_chosen = np.ones(num_actions)
        self.sum_rewards = np.zeros(num_actions)

    def update(self, action, reward):
        # self.num_actions_chosen[action] += 1
        self.sum_rewards[action] += reward

    def reset(self):
        self.num_actions_chosen = np.ones(self.num_actions)
        self.sum_rewards = np.zeros(self.num_actions)

    def choose_action(self, return_prob=False, behavior_policy=True, chosen_action=None):

        best_action = np.argmax(self.sum_rewards / self.num_actions_chosen)

        probabilities = [self.epsilon / (self.num_actions - 1)] * self.num_actions
        probabilities[best_action] = 1 - self.epsilon

        if return_prob:

            if not behavior_policy:
                self.num_actions_chosen[chosen_action] += 1
            return probabilities

        chosen_action = np.random.choice(list(range(self.num_actions)), p=probabilities)

        self.num_actions_chosen[chosen_action] += 1

        return chosen_action, probabilities[chosen_action]

class UCB1:
    def __init__(self, num_actions, reset_period=10, c=2):
        self.policy_type = 'M'
        self.n = 1
        self.reset_period = reset_period
        self.c = c
        self.num_actions = num_actions
        self.k_n = np.ones(num_actions)
        self.sum_rewards = np.zeros(num_actions)

    def update(self, action, reward):
        # self.num_actions_chosen[action] += 1
        self.sum_rewards[action] += reward

    def reset(self):
        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.num_actions)
        self.sum_rewards = np.zeros(self.num_actions)

    def choose_action(self, return_prob=False, behavior_policy=True, chosen_action=None):
        a = np.argmax(self.sum_rewards / self.k_n + self.c * np.sqrt(np.log(self.n) / self.k_n))

        if return_prob:
            probabilities = [0] * self.num_actions
            probabilities[a] = 1

            if not behavior_policy:
                self.n += 1
                self.k_n[chosen_action] += 1

            return probabilities

        self.n += 1
        self.k_n[a] += 1

        if self.n % self.reset_period == 0:
            self.reset()

        return a, 1
