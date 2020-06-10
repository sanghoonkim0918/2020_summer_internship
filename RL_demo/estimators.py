class Evaluation:
    def __init__(self, horizon, num_target_policies, num_actions):
        """
        inputs
        :param horizon: size of horizon (number of requests)
        :param num_target_policies: number of target policies
        :param num_actions: number of actions avilable (number of servers)
        """
        self.horizon = horizon
        self.num_target_policies = num_target_policies
        self.num_actions = num_actions
        self.num_trajectories = 0 # number of trajectories received so far

        """
        Implement IS, stepwise IS, WIS, stepwise WIS
        """