import numpy as np
from math import ceil

class Servers:
    def __init__(self, num_policies, num_servers):
        """
        num_policies: number of policies we want to run
        num_servers: number of servers for each policy
        """
        self.num_policies = num_policies
        self.num_servers = num_servers
        self.server_load = {policy_index: np.zeros(num_servers) for policy_index in range(num_policies)}

    def get_latency(self, policy_index, chosen_server):
        """
        return the latency of the request sent to the chosen server.
        The latency is sampled from a lognormal distribution, which is
        shifted to the right by the server load of the chosen server. 
        """
        chosen_server_load = self.server_load[policy_index][chosen_server]

        latency = chosen_server_load + np.random.lognormal(0, 0.5)

        if latency < 1:
            latency = 1

        timer = ceil(latency)

        return latency, timer

    def reset(self):
        """
        After one epsiode is finished, set every sever load to be 0.
        """
        self.server_load = {policy_index: np.zeros(self.num_servers) for policy_index in range(self.num_policies)}
