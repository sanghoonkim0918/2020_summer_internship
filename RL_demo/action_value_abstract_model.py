class ActionValueModel:
    r"""
    ActionValueModel is a class that represents
    an action value model (Q_\pi) for a fixed (stationary) policy \pi
    in reinforcement learning (RL) framework.

    You can have any action value model (DM), but it should subclass
    this abstract class.
    """

    def __init__(self, horizon, num_actions, policy):
        """
        Params:
            horizon: horizon of the environment (We use a fixed horizon here)
            num_actions: number of actions (we are not using continus actions; action index starts from 0)
            policy: the policy that we want to learn the Q value of
        """

        self.horizon = horizon
        self.num_actions = num_actions
        self.policy = policy

    def get_Q(self, index, state, action):
        """
        Return Q_pi(state, action) at the given index

        Params:
            index: the step of Q_pi we want to get (thus, index ranges from 0 to self.horizon - 1)
            state: the state in Q_pi(state, action)
            action: the action in Q_pi(state, action)
        """

        raise NotImplementedError

    def get_V(self, index, sample):
        """
        Return V(state)

        Params:
            index: the step of Q_pi we want to get (thus, index ranges from 0 to self.horizon - 1)
            sample: the sample of the given index from a trace.
                    sample is a list (or tuple) of [state, action, reward, behavior_probabilities]
        """

        if index > self.horizon - 1:
            return 0

        state = sample[0]
        V_estimate = 0
        for action_index, new_prob in enumerate(self.policy.choose_action(context=state,
                                                                          return_prob=True,
                                                                          behavior_policy=False,
                                                                          chosen_action=sample[1])):
            V_estimate += new_prob * self.get_Q(index, state, action_index)

        return V_estimate

    def update_Q(self, index, sample, next_sample):
        """
        Update Q_\pi at the given index, with the given next state.

        Params:
            index: the step of Q_pi we want to update (thus, index ranges from 0 to self.horizon - 1)
            sample: the sample of the given index from a trace.
                    sample is a list (or tuple) of [state, action, reward, behavior_probabilities]
            next_sample: the next sample observed after the given sample in the trace.
        """

        raise NotImplementedError