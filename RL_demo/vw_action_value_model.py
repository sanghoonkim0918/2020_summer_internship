from action_value_abstract_model import ActionValueModel
from vowpalwabbit import pyvw

class VWActionValueModel(ActionValueModel):
    r"""
    An ActionValueModel using VowpalWabbit (VW)

    the given VW format can be changed as you want.
    Think about if the given VW format is correct! (This is not a trick question)
    """

    def __init__(self, horizon, num_actions, policy, default_model="--power_t 0.0 -q la --quiet"):
        """
        Initialize variables to store basic information of MDP
        and every Q model needed for each step.

        Params:
            default_model: a default VW model to learn Q. You can try different settings to get the best model.
        """
        super().__init__(horizon, num_actions, policy)

        # We assume that each step is indexed from 1 to H (horizon)
        self.models = [pyvw.vw(default_model) for _ in range(self.horizon)]

    def vw_format(self, state_server_load, action, reward=None):
        """
        Convert a given sample to a VW example.
        The current format of state only includes server load. Change it if needed.

        Params:
            state_server_load: the state in the given sample.
            action: the action chosen by the logging policy in the given sample
            reward: the reward revealed by the chosen action.
                    If the function vw_format is used to predict Q value,
                    reward should be None.
        """

        vw_example = ""

        if reward is not None:
            vw_example += f"{reward} "

        # Add server load (l stands for load)
        vw_example += "|l"

        for server_index in range(self.num_actions):
            vw_example += f" load{server_index}:{state_server_load[server_index]}"

        # Add the chosen action
        vw_example += f" |a server{action}"

        return vw_example

    def get_Q(self, index, state, action):
        """
        Return Q_pi(state, action) at the given index

        Params:
            index: the step of Q_pi we want to get (thus, index ranges from 0 to self.horizon - 1)
            state: the state in Q_pi(state, action).
            action: the action in Q_pi(state, action)
        """
        if index > self.horizon - 1:
            return 0

        model = self.models[index]
        sample_in_vw_format = self.vw_format(state_server_load=state[0],
                                             action=action)

        return model.predict(sample_in_vw_format)

    def update(self, index, sample, next_sample):
        """
        Update Q_\pi at the given index, with the given next state.

        Params:
            index: the step of Q_pi we want to update (thus, index ranges from 0 to self.horizon - 1)
            sample: the sample of the given index from a trace.
                    sample is a list (or tuple) of [state, action, reward, behavior_probabilities]
            next_sample: the next sample observed after the given sample in the trace.
        """
        next_state_V_estimate = self.get_V(index + 1, next_sample)

        sample_in_vw_format = self.vw_format(state_server_load=sample[0][0],
                                             action=sample[1],
                                             reward=sample[2] + next_state_V_estimate)

        model = self.models[index]
        model.learn(sample_in_vw_format)


