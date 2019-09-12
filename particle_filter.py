import numpy as np


class ParticleFilter:
    def __init__(self, data, states, particles, times, trans_prob):
        self.data = data
        self.states = states
        self.times = times
        self.trans_prob = trans_prob

    def run(self):
        x = np.random.randint(states, size=particles)
        max_prob = np.zeros(time, dtype=np.int)
        for t in range(times):
            post = np.zeros(states)
            for i in range(particles):
                x[i] = np.random.choice(states, p=trans_prob[i])
                weight[i] = calculate_weight(data, x[i])
                post[x[i]] += weight[i]
            if weight.sum() > 0:
                weight /= weight.sum()
                max_prob[t] = post.argmax()
                resample = np.random.choice(states, particles, p=weight)
                x = x[resample]
            else:
                max_prob[t] = max_prob[t-1]


def calculate_weight(data, x):
    pass
