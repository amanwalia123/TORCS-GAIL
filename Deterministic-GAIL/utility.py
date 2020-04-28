from scipy import signal
import numpy as np

def calc_disc_return(rewards,gamma):
    """
        calculate discounted return for the episode
    """
    rewards = np.array(rewards)
    disc_return = signal.lfilter([1], [1, float(-gamma)], rewards[::-1], axis=0)[::-1]
    return disc_return
