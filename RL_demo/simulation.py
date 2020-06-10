import numpy as np
from policies import *
from environment import Servers

def run_simulation(policies, num_requests, servers):
    """
    state_transition_noise is not implemented

    :param policies:
    :param policy_names:
    :param num_requests:
    :param num_servers:
    :param noise_type:
    :return:
    """
    # Variables initialization
    num_policies = len(policies)
    termination = False
    behavior_policy_index = num_policies - 1 # behavior policy should be the last element of the list policies

    # Initialize requests
    request_index = 0
    request_sizes = [1]
    requests = list(np.random.choice(request_sizes, size=num_requests))

    # Initialize store variables
    requests_being_processed_for_each_policy = [list() for _ in range(num_policies)]
    total_latency_for_each_policy = np.zeros(num_policies)

    # trace is a list of lists (samples) of [state, action, reward, behavior_probabilities]
    # the current form of state is: [server_load, list of (action, reward) revealed in the current time step]
    trace = [[[None, list()], None, None, None] for _ in range(num_requests)]

    # Helper functions
    def timer_pass(policy_index):
        """
        Check if there are requests that are completely processed.
        Server spits out those requests. 
        """

        timer_out = []
        indices = []

        requests_being_processed = requests_being_processed_for_each_policy[policy_index]

        for arbitrary_request_index in range(len(requests_being_processed)):
            request_info = requests_being_processed[arbitrary_request_index]

            timer = request_info['timer']

            if timer == 0 or timer == 1:
                indices.append(arbitrary_request_index)
            else:
                requests_being_processed[arbitrary_request_index]['timer'] -= 1

        for index in sorted(indices, reverse=True):
            timer_out.append(requests_being_processed.pop(index))

        timer_out.reverse()

        return timer_out

    # Start the simulation
    while not termination:

        # Get a request
        if requests:
            request = requests.pop()
        else:
            request = None

        for policy_index in range(num_policies):
            # Store the name of the current policy
            policy = policies[policy_index][1]

            # Reward revealed
            timer_out = timer_pass(policy_index)

            if timer_out:
                for request_info in timer_out:
                    chosen_server = request_info['chosen_server']
                    reward = request_info['reward']

                    # Record the reward revealed
                    total_latency_for_each_policy[policy_index] += reward

                    # Take out the request of the reward from the server
                    servers.server_load[policy_index][chosen_server] -= request_info['request_size']

                    # Update the policy
                    policy.update(chosen_server, reward)

                    # If the policy is the behavior policy
                    if policy_index == behavior_policy_index:  # Behavior policy should be the first policy (index: 0) in the tuple policies
                        revealed_request_index = request_info['request_index']

                        # For the "corresponding" request, record the revealed reward with the corresponding action
                        trace[revealed_request_index][1] = chosen_server
                        trace[revealed_request_index][2] = reward

                        # For the "current" state, record the revealed reward with the corresponding action.
                        # This is for learning of target policies while doing counterfactual evaluations
                        if request_index < num_requests:
                            # This is not needed if the reward is revealed after every request is sent
                            trace[request_index][0][2].append((chosen_server, reward))

            if request is not None:
                if policy_index == behavior_policy_index:  # Behavior policy should be the first policy (index: 0) in the tuple policies
                    # record the probability of the behavior policy choosing each action 
                    behavior_probabilities = policy.choose_action(context=servers.server_load[policy_index], return_prob=True)
                    trace[request_index][3] = behavior_probabilities

                    # Record the current server load, which is a part of state, for evaluation
                    server_load = servers.server_load[policy_index]
                    context = np.empty_like(server_load)
                    np.copyto(context, server_load)
                    trace[request_index][0][0] = context

                # Policy chooses an action
                chosen_server, _ = policy.choose_action(context=servers.server_load[policy_index])

                latency_of_request, timer = servers.get_latency(policy_index, chosen_server)

                # Record the current request to be routed
                request_info = {'reward': -latency_of_request, 'timer': timer,
                                'chosen_server': chosen_server,
                                'request_index': request_index, 'request_size': request}


                # Increment the server load by the size of the request after the request is sent
                servers.server_load[policy_index][chosen_server] += request

                # Append the routed request to the list of the requests being processed of the policy
                requests_being_processed_for_each_policy[policy_index].append(request_info)


        # Increment the request index
        request_index += 1

        # Check if the simulation is finished
        if not requests:  # no more request to process
            simulation_done = True

            for requests_being_processed in requests_being_processed_for_each_policy:
                if requests_being_processed:
                    simulation_done = False
                    break

            termination = simulation_done

    # Reset server and every policy
    servers.reset()
    for policy_index in range(num_policies):
        policies[policy_index][1].reset()

    return total_latency_for_each_policy, trace


if __name__ == "__main__":
    num_requests = 20
    num_servers = 4

    np.random.seed(0)

    behavior_policy = ('EpsilonGreedy_behavior', EpsilonGreedy(num_servers, epsilon=0.3))
    target_policy1 = ('UCB1_target1', UCB1(num_servers, reset_period=20))
    policies = [target_policy1, behavior_policy]

    servers = Servers(num_policies=len(policies), num_servers=num_servers, noise_type='increasing_variance')

    run_simulation(policies, num_requests, servers)
