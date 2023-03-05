import numpy as np

MU = np.array([0.3, 0.28, 0.1])

def main(n_sample):
    n_arms = len(MU)
    # hyper parameter
    success = np.ones(n_arms)
    failure = np.ones(n_arms)
    for _ in range(n_sample):
        proba = [
            np.random.beta(success[arm], failure[arm])
            for arm in range(n_arms)
        ]
        arm = np.argmax(proba)
        if np.random.rand() < MU[arm]:
            success[arm] += 1
        else:
            failure[arm] += 1
    print('count selection of arms')
    for arm in range(n_arms):
        print('  true proba =', MU[arm], ':')
        print('    count :', success[arm] + failure[arm])
        print('    proba :', success[arm] / (success[arm] + failure[arm]))


if __name__ == "__main__":
    main(100000)
