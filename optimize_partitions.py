import numpy as np
import qfl
from nfl_strategy import NFLStrategy
from test_qfl import game_parameters
from itertools import product
from typing import Tuple
import time

# State_space: (min_(downs left/yards_to_reset_downs), max_(downs left/yards_to_reset_downs), min_(yards to score / time left), max_(yards to score / time left))
TIME_LIMIT = 9
GAMES_LIMIT = 100

def optimize_specific_partition():
    x_candidates = np.linspace(2, 5, 10)
    y_candidates = np.linspace(2, 5, 10)
    best_partitions, best_partitions_score = None, 0
    model = NFLStrategy(*game_parameters[0])

    for x1 in x_candidates:
        for x2 in x_candidates:
            if x2 <= x1:
                continue
            for y1 in y_candidates:
                for y2 in y_candidates:
                    if y2 <= y1:
                        continue
                    partitions = ((x1, x2), (y1, y2))
                    policy = qfl.q_learn(model, TIME_LIMIT, partitions)
                    score = model.simulate(policy, GAMES_LIMIT)
                    if score > best_partitions_score:
                        best_partitions, best_partitions_score = partitions, score

    return best_partitions, best_partitions_score
    
    
      
if __name__ == '__main__':
    # (x_dimension_splits: (yards_to_score/time_left1, yards_to_score/time_left2), y_dimension_splits: (downs_left/yards_to_reset_downs1, downs_left/yards_to_reset_downs2))
    optimum_partitions, optimum_score = optimize_specific_partition()
    print(f'Score: {optimum_score}; Optimum Partitions: x: ({optimum_partitions[0][0], optimum_partitions[0][1]}), y: ({optimum_partitions[1][0], optimum_partitions[1][1]})')
        