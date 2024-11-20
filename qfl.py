from nfl_strategy import NFLStrategy
import time
from typing import Tuple
from collections import defaultdict
import random
from math import exp
from typing import List
from pprint import pprint

# Parameters
EPSILON = 0.175
ALPHA = 0.25
MIN_ALPHA = 0.01
GAMMA = 1
LAMBDA = 0.995

def initialize_q_table(playbook_size: int) -> List[List[List[float]]]:
    """
        Initialize 3X3 Q-table with zeros.
            playbook_size: number of plays in the offensive playbook
    """
    return [[[0 for _ in range(playbook_size)] for _ in range(3)], [[0 for _ in range(playbook_size)] for _ in range(3)], [[0 for _ in range(playbook_size)] for _ in range(3)]]
    

def find_partition(state: Tuple[int, int, int, int], divisions: Tuple[Tuple[float, float], Tuple[float, float]])  -> Tuple[int, int]:
    """
        Return the partition that the given state belongs to.
            state: [remaining_yards_to_score, downs_left, yards_to_reset_downs, time_left(5-second ticks)]
            divisions: (x_dimension_splits: (yards_to_score/time_left1, yards_to_score/time_left2), y_dimension_splits: (downs_left/yards_to_reset_downs1, downs_left/yards_to_reset_downs2))
    """
    x = state[0] / state[3]   # yards_to_score/time_left
    y = state[1] / state[2] # downs_left/yards_to_reset_downs
    x_dimension_boundaries = divisions[0]
    y_dimension_boundaries = divisions[1]
    if x <= x_dimension_boundaries[0]:
        state_x = 0
    elif x <= x_dimension_boundaries[1]:
        state_x = 1
    else:
        state_x = 2

    if y <= y_dimension_boundaries[0]:
        state_y = 0
    elif y <= y_dimension_boundaries[1]:
        state_y = 1
    else:
        state_y = 2
    return (state_x, state_y)

def epsilon_greedy_action(q: List[List[List[float]]], partition: Tuple[int, int], model: NFLStrategy) -> int:
    """
        Return the best action for the given state using the epsilon-greedy policy.
            q: Q-values for each partition-action pair
            state: [remaining_yards_to_score, downs_left, yards_to_reset_downs, time_left(5-second ticks)]
            epsilon: probability of selecting a random action
    """
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, model.offensive_playbook_size() - 1)
    else:
        x, y = partition
        action, value_of_action = max(enumerate(q[x][y]), key=lambda x: x[1])
        # Choose random action if all actions have same 0 value
        return action if value_of_action else random.randint(0, model.offensive_playbook_size() - 1)
    
def calculate_partition_boundaries(model, n_episodes=1000):
    """Sample states to determine more balanced partition boundaries"""
    states = []
    for _ in range(n_episodes):
        state = model.initial_position()
        while not model.game_over(state):
            states.append(state)
            action = random.randint(0, model.offensive_playbook_size() - 1)
            state, _ = model.result(state, action)
    
    x_values = sorted([s[0]/s[3] for s in states])
    y_values = sorted([s[1]/s[2] for s in states])
    n = len(states)
    
    return (
        (x_values[n//3], x_values[2*n//3]),
        (y_values[n//3], y_values[2*n//3])
    )

def qfl_policy(state: Tuple[int, int, int, int], model: NFLStrategy, time_limit: float) -> int:
    """
        Return the best action for the given state using the q-learning QFL policy.
            state: [remaining_yards_to_score, downs_left, yards_to_reset_downs, time_left(5-second ticks)]
            model: NFLStrategy object
            time_limit: time limit to evaluate policy
    """
    start = time.time()
    q = initialize_q_table(model.offensive_playbook_size())
    alpha_values = defaultdict(lambda: ALPHA) # Seperate alpha values for each partition; #9 is for terminal states
    partitions_stats = defaultdict(int)
    state_space_partitions = ((3.1, 3.3), (0.3, 0.4))
    # state_space_partitions = calculate_partition_boundaries(model)
    # print(state_space_partitions)
    current_state, episode = state, 0
    while (time.time() - start) < time_limit and not model.game_over(current_state):
        partition = find_partition(current_state, state_space_partitions)
        partition_x, partition_y = partition
        action = epsilon_greedy_action(q, partition, model)
        next_state, reward = model.result(current_state, action)
        if model.game_over(next_state):
            # Terminal state
            reward = 1 if model.win(next_state) else -1
            alpha_values[(-1, -1)] = max(alpha_values[(-1, -1)] * exp(-LAMBDA * episode), MIN_ALPHA)
            q[partition_x][partition_y][action] += alpha_values[(-1, -1)] * (reward - q[partition_x][partition_y][action]) # separate partition for terminal states
        else:
            # Non-terminal state
            next_state_partition_x, next_state_partition_y = find_partition(next_state, state_space_partitions)
            reward = reward[0] / 100 # Reward progress towards scoring
            next_q = max(q[next_state_partition_x][next_state_partition_y])
            alpha_values[partition] = max(alpha_values[partition] * exp(-LAMBDA * episode), MIN_ALPHA)
            q[partition_x][partition_y][action] += alpha_values[partition] * ((reward + (GAMMA * next_q)) - q[partition_x][partition_y][action])
        current_state = next_state
        partitions_stats[partition] += 1
        episode += 1
    # pprint(partitions_stats)
    x, y = find_partition(state, state_space_partitions)
    return max(enumerate(q[x][y]), key=lambda x: x[1])[0]

            
def q_learn(model: NFLStrategy, time_limit: float) -> callable:
    return lambda state: qfl_policy(state, model, time_limit)