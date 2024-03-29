import numpy as np

MU = np.array([3.0, 2.0, 1.0])
SIGMA = 10.0


def main(n_sample: int):
    # setting
    n_item = len(MU)
    n_user = 100
    n_attribute = 5
    sigma_0 = 1.0
    theta_attribute = np.random.randn(n_attribute)
    theta_true = np.concatenate([MU, theta_attribute])
    user_attribute = np.random.randn(n_user, n_attribute)

    # initialization
    a_inv = sigma_0 / SIGMA * np.eye(len(theta_true))
    b = np.zeros(len(theta_true))

    # for record
    result = [[] for _ in range(n_item)]
    for _ in range(n_sample):
        user = np.random.randint(n_user)
        attribute = user_attribute[user]
        mu = a_inv @ b
        gamma = SIGMA * a_inv
        gamma_sqrt = np.linalg.cholesky(gamma)
        theta_sample = mu + gamma_sqrt @ np.random.randn(len(mu))
        action = np.hstack([np.eye(n_item), attribute[np.newaxis, :].repeat(n_item, axis=0)])
        value = action @ theta_sample
        arm = np.argmax(value)
        reward = action[arm] @ theta_true + SIGMA * np.random.randn()

        # update param
        tmp = a_inv @ action[arm]
        a_inv -= np.outer(tmp, tmp) / (1.0 + tmp @ action[arm])
        b += action[arm] * reward

        # record result
        result[arm].append(reward)
    print('count selection of arms')
    for item in range(n_item):
        print('  item', item, '|| N(', MU[item], ',', SIGMA, '):')
        print('    count :', len(result[item]))
        if result[item]:
            print('    mean  :', np.mean(result[item]))
            print('    stdev :', np.std(result[item]))
    regret = n_sample * MU.max() - sum(sum(r) for r in result)
    print('regret:', regret)
    print('true param:', theta_true)
    print('pred param:', a_inv @ b)


if __name__ == "__main__":
    main(100000)
