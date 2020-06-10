from simulation import run_simulation
import numpy as np
from policies import *
from environment import Servers
from estimators import Evaluation
import sys

if __name__ == "__main__":
    # Initialize basic variables
    num_seeds = 5#int(input("num_seeds: "))
    num_requests = 15#int(input("num_requests: "))
    num_servers = 4#int(input("num_servers: "))
    threshold = int(sys.argv[1])#int(input("threshold: "))
    num_trajectories_list = [0] + [10 ** i for i in range(1, 7)]
    start_trajectory_index = 0
    # end_trajectory_index = 1
    noise_type = 'increasing_variance'

    # Initialize policies
    behavior_policy = ('UniRan', UniRan(num_servers))
    target_policy1 = ('LeastLoad_0.3', LeastLoad(num_servers, epsilon=0.3))
    target_policy2 = ('LeastLoad_0', LeastLoad(num_servers, epsilon=0))
    target_policy3 = ('EpsilonGreedy', EpsilonGreedy(num_servers, epsilon=0.3))
    target_policy4 = ('UCB1', UCB1(num_servers))

    # behavior_policy = ('EpsilonGreedy', EpsilonGreedy(num_servers, epsilon=0.3))
    # target_policy1 = ('LeastLoad_0.3', LeastLoad(num_servers, epsilon=0.3))
    # target_policy2 = ('LeastLoad_0', LeastLoad(num_servers, epsilon=0))
    # target_policy3 = ('UniRan', UniRan(num_servers))
    # target_policy4 = ('UCB1', UCB1(num_servers))


    target_policies = [target_policy1, target_policy2, target_policy3, target_policy4]
    num_target_policies = len(target_policies)

    policies = [target_policy1, target_policy2, target_policy3, target_policy4, behavior_policy]
    num_policies = 1 + num_target_policies

    target_policy_names = [policy[0] for policy in policies if policy != policies[-1]]

    # Initialize environment (servers)
    servers = Servers(num_policies=num_policies, num_servers=num_servers)

    # Initialize evaluation for each seed
    evaluations = [Evaluation(horizon=num_requests,
                              num_target_policies=num_target_policies,
                              num_actions=num_servers)
                   for _ in range(num_seeds)]

    # Write output to a file
    file_name = f"num_requests_{num_requests}_num_servers_{num_servers}_threshold_{threshold}_no_noise_change.txt" #input("Enter file name: ")
    with open(file_name, 'w') as wf:
        # Iterate over number of trajectories
        for num_trajectories_index in range(1, len(num_trajectories_list)):
            write_string = f"\nnum_seeds: {num_seeds}" + f"\nnum_trajectory: {num_trajectories_list[num_trajectories_index]}" \
                           + f"\nnum_requests: {num_requests}" \
                           + f"\nnum_servers: {num_servers}" + f"\nbehavior policy: {behavior_policy[0]}" \
                           + f"\ntarget policies: {target_policy_names}"
            print(write_string)
            wf.write(write_string)

            # Iterate over random seed
            seed_add = 0
            for seed in range(seed_add, seed_add + num_seeds):
                # Set a numpy random seed
                np.random.seed(seed)

                # Set a particular evaluation for the seed
                seed_evaluation = evaluations[seed - seed_add]

                # Iterate over trajectories of the given number to run simulation
                for trajectory_index in range(num_trajectories_list[num_trajectories_index - 1], num_trajectories_list[num_trajectories_index]):
                    print(f"seed:{seed - seed_add} / trajectory: {seed_evaluation.num_trajectories}", end='\r')

                    # Run simulation and get trace & true performances of every target policy
                    total_latency_for_each_policy, trace = \
                        run_simulation(policies=policies, num_requests=num_requests, servers=servers)

                    # Get the estimate of every target policy on the trajectory. 
                    """
                    Implement
                    """

            # After finishing iterating over every random seed,
            # display the estimate of each target policy for the given number of trajectories
            for seed_evaluation in evaluations:
                """
                Implement
                """




