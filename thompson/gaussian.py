import numpy as np

MU = np.array([3.0, 2.0, 1.0])
SIGMA = np.array([60.0, 40.0, 20.0])


def update_parameters(x, mu, gamma, alpha, beta):
    alpha += 0.5
    beta += 0.5 * (x * x + gamma * mu * mu)
    gamma += 1.0
    mu = ((gamma - 1.0) * mu + x) / gamma
    beta -= 0.5 * gamma * mu * mu
    return mu, gamma, alpha, beta


def gaussian_thompson_sampling(mu, gamma, alpha, beta):
    t = np.random.standard_t(2 * alpha)
    value = t * np.sqrt(beta / alpha / gamma) + mu
    return value


def main(n_sample):
    n_arms = len(MU)
    # hyper parameters
    alpha = -np.ones(n_arms) # Degree of freedom
    beta = np.zeros(n_arms) # Variance times half of sample size
    gamma = np.zeros(n_arms) # Sample size
    mu = np.zeros(n_arms) # Average
    # for record
    result = [[] for _ in range(n_arms)]
    for arm in range(n_arms):
        for _ in range(3):
            x = np.random.randn() * SIGMA[arm] + MU[arm]
            mu[arm], gamma[arm], alpha[arm], beta[arm] = (
                update_parameters(x, mu[arm], gamma[arm], alpha[arm], beta[arm])
            )
            result[arm].append(x)
    for _ in range(n_sample - 3 * n_arms):
        values = [
            gaussian_thompson_sampling(mu, gamma, alpha, beta)
            for mu, gamma, alpha, beta in zip(mu, gamma, alpha, beta)
        ]
        arm = np.argmax(values)
        x = np.random.randn() * SIGMA[arm] + MU[arm]
        mu[arm], gamma[arm], alpha[arm], beta[arm] = (
            update_parameters(x, mu[arm], gamma[arm], alpha[arm], beta[arm])
        )
        result[arm].append(x)
    print('count selection of arms')
    for arm in range(n_arms):
        print('  arm', arm, '|| N(', MU[arm], ',', SIGMA[arm] ** 2, '):')
        print('    count :', len(result[arm]))
        print('    mean  :', np.mean(result[arm]))
        print('    stdev :', np.std(result[arm]))
    regret = n_sample * MU.max() - sum(sum(r) for r in result)
    print('regret:', regret)


if __name__ == "__main__":
    main(100000)
