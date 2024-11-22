from nfl_strategy import NFLStrategy
import time
from typing import Tuple
from collections import defaultdict
import random
from math import exp
from typing import List

# Parameters
EPSILON = 0.25
MIN_EPSILON = 0.1
EPISODES_TO_MIN = 1000
ALPHA = 0.25
MIN_ALPHA = 0.05
GAMMA = 1
LAMBDA = 0.997

def initialize_q_table(playbook_size: int) -> List[List[List[float]]]:
    """
        Initialize 3X3 Q-table with zeros.
            playbook_size: number of plays in the offensive playbook
    """
    return [[[0 for _ in range(playbook_size)] for _ in range(3)], [[0 for _ in range(playbook_size)] for _ in range(3)], [[0 for _ in range(playbook_size)] for _ in range(3)]]
    

def find_partition(state: Tuple[int, int, int, int])  -> Tuple[int, int]:
    """
        Return the partition that the given state belongs to.
            state: [remaining_yards_to_score, downs_left, yards_to_reset_downs, time_left(5-second ticks)]
            divisions: (x_dimension_splits: (yards_to_score/time_left1, yards_to_score/time_left2), y_dimension_splits: (downs_left/yards_to_reset_downs1, downs_left/yards_to_reset_downs2))
    """
    divisions = ((2.0, 5.0), (2.0, 3.6))
    x = state[0] / state[3]   # yards_to_score/time_left
    y = state[2] / state[1] # yards_to_reset_downs/downs_left
    x_dimension_boundaries = divisions[0]
    y_dimension_boundaries = divisions[1]
    state_x = 0 if x <= x_dimension_boundaries[0] else (1 if x <= x_dimension_boundaries[1] else 2)
    state_y = 0 if y <= y_dimension_boundaries[0] else (1 if y <= y_dimension_boundaries[1] else 2)
    return (state_x, state_y)

def epsilon_greedy_action(q: List[List[List[float]]], partition: Tuple[int, int], model: NFLStrategy, episode: int) -> int:
    """
        Return the best action for the given state using the epsilon-greedy policy.
            q: Q-values for each partition-action pair
            state: [remaining_yards_to_score, downs_left, yards_to_reset_downs, time_left(5-second ticks)]
            epsilon: probability of selecting a random action
    """
    decay = max(0, (EPISODES_TO_MIN - episode) / EPISODES_TO_MIN)
    if random.uniform(0, 1) < max(EPSILON * decay, MIN_EPSILON):
        return random.randint(0, model.offensive_playbook_size() - 1)
    else:
        x, y = partition
        return max(enumerate(q[x][y]), key=lambda x: x[1])[0]

def q_learn(model: NFLStrategy, time_limit: float) -> callable:
    """
        Return the best action for the given state using the q-learning QFL policy.
            state: [remaining_yards_to_score, downs_left, yards_to_reset_downs, time_left(5-second ticks)]
            model: NFLStrategy object
            time_limit: time limit to evaluate policy
    """
    q = initialize_q_table(model.offensive_playbook_size())
    start = time.time()
    alpha_values = defaultdict(lambda: ALPHA) # Separate alpha values for each partitions
    partition_stats = defaultdict(int)
    while (time.time() - start) < time_limit:
        current_state, episode = model.initial_position(), 0
        while not model.game_over(current_state):
            partition = find_partition(current_state)
            partition_x, partition_y = partition
            action = epsilon_greedy_action(q, partition, model, episode)
            next_state, reward = model.result(current_state, action)
            alpha_values[partition] = max(alpha_values[partition] * exp(-LAMBDA * partition_stats[partition]), MIN_ALPHA)
            if model.game_over(next_state):
                # Terminal state
                reward = 10 if model.win(next_state) else -10
                q[partition_x][partition_y][action] += alpha_values[partition] * (reward - q[partition_x][partition_y][action]) 
            else:
                # Non-terminal state
                next_state_partition_x, next_state_partition_y = find_partition(next_state)
                next_q = max(q[next_state_partition_x][next_state_partition_y])
                q[partition_x][partition_y][action] += alpha_values[partition] * (reward[0]/100 + (GAMMA * next_q) - q[partition_x][partition_y][action])
            current_state = next_state
            episode += 1
            partition_stats[partition] += 1
    # print(partition_stats)
    return lambda state: max(enumerate(q[find_partition(state)[0]][find_partition(state)[1]]), key=lambda x: x[1])[0]