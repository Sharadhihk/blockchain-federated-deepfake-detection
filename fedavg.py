import numpy as np

def federated_average(weight_list):

    avg_weights = []

    for weights in zip(*weight_list):

        avg_weights.append(
            np.mean(weights, axis=0)
        )

    return avg_weights