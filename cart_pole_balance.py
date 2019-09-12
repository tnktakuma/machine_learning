from math import sin, cos
from random import gauss
import numpy as np

TAU = 1 / 60
G = 9.8


def sgn(x: float) -> float:
    if x > 0.:
        return 1.
    elif x < 0.:
        return -1.
    else:
        return 0.


def normalize(v: np.ndarray) -> np.ndarray:
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return np.zeros_like(v)
    else:
        return v / v_norm


class CartPole:
    """Cart and Pole

    args:
        M (float): Mass of Cart
        m (float): Mass of Pole
        l (float): Half length of Pole
        mu_c (float): Friction Coefficient of Cart
        mu_p (float): Friction Coefficient of Pole
    """
    def __init__(self, M=1.0, m=0.1, l=0.5, mu_c=5e-4, mu_p=2e-6):
        self.cart_pos = gauss(0., 0.01)
        self.cart_vel = gauss(0., 0.01)
        self.pole_ang = gauss(0., 0.01)
        self.pole_vel = gauss(0., 0.01)
        self.M = M
        self.m = m
        self.l = l
        self.mu_c = mu_c
        self.mu_p = mu_p

    def get_state(self) -> np.ndarray:
        return np.array([self.cart_pos, self.cart_vel, self.pole_ang, self.pole_vel])

    def elapse(self, action: float):
        pole_acc = self.pole_acceleration(action)
        cart_acc = self.cart_acceleration(action, pole_acc)
        self.cart_pos += self.cart_vel * TAU
        self.cart_vel += cart_acc * TAU
        self.pole_ang += self.pole_vel * TAU
        self.pole_vel += pole_acc * TAU

    def pole_acceleration(self, action: float) -> float:
        moi = self.m * self.l
        mass = self.M + self.m
        gravity = G * sin(self.pole_ang)
        interaction = self.mu_c * sgn(self.cart_vel) - action
        interaction -= moi * self.pole_vel * self.pole_vel * sin(self.pole_ang)
        interaction *= cos(self.pole_ang) / mass
        fraction = self.mu_p * self.pole_vel / moi
        denom = 4/3 - self.m * cos(self.pole_ang) * cos(self.pole_ang) / mass
        return (gravity + interaction - fraction) / (denom * self.l)

    def cart_acceleration(self, action: float, pole_acc: float) -> float:
        interaction = self.pole_vel * self.pole_vel * sin(self.pole_ang)
        interaction -= pole_acc * cos(self.pole_ang)
        interaction *= self.m * self.l
        fraction = self.mu_c * sgn(self.cart_vel)
        return (action + interaction - fraction) / (self.M + self.m)


class Agent:
    def __init__(self, episodes=1000, C=[5/12, 0.5, 15/np.pi, 2/3],
        Q=[1.25, 1., 12., 0.25], R=0.01, alpha=0.1, gamma=0.95, epsilon=3e-3):
        self.episodes = episodes
        self.C = np.array(C)
        self.Q = np.array(Q)
        self.R = R
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.theta = np.random.rand(4) * 10. - 5.
        self.eta = np.random.rand(1) * 2. - 1.

    def get_action(self, state: np.ndarray, restrict=20.) -> float:
        self.mu = self.theta @ (self.C * state)
        self.sigma = 0.1 + 1. / (1. + np.exp(self.eta))
        action = gauss(self.mu, self.sigma)
        if action > restrict:
            return restrict
        elif action < -restrict:
            return -restrict
        else:
            return float(action)

    def get_gradient(self, action: float, state: np.ndarray) -> np.ndarray:
        theta_grad = (action - self.mu) / (self.sigma ** 2) * self.C * state
        eta_grad = (action - self.mu) ** 2 / (self.sigma ** 3) - 1 / self.sigma
        eta_grad *= (self.sigma - 0.1) * (0.9 - self.sigma)
        return np.append(theta_grad, eta_grad)

    def start_episode(self, max_time=60.0):
        cp = CartPole()
        state = cp.get_state()
        time = 0.
        z = np.zeros(5)
        delta = np.zeros(5)
        discount = 1.
        total_reward = 0.
        while np.all(np.abs(self.C * state) < 1.) and time < max_time:
            action = self.get_action(state)
            reward = -state @ (self.Q * state) - self.R * action ** 2
            z += self.get_gradient(action, state)
            delta += discount * reward * z
            total_reward += discount * reward
            discount *= self.gamma
            cp.elapse(action)
            state = cp.get_state()
            time += TAU
        print(f'time={time:06.3f}, R={total_reward:.3e}', end=' ')
        return time, total_reward, delta

    def train(self):
        times = []
        total_rewards = []
        n = 0
        cur_delta = np.zeros(5)
        for ep in range(self.episodes):
            print(f'\r{ep + 1} / {self.episodes} :', end=' ')
            pre_delta = cur_delta
            time, total_reward, delta = self.start_episode()
            times.append(time)
            total_rewards.append(total_reward)
            cur_delta = (n * cur_delta + time) / (n + 1)
            n += 1
            if normalize(pre_delta) @ normalize(cur_delta) > cos(self.epsilon):
                self.theta += self.alpha * cur_delta[:4]
                self.eta += self.alpha * cur_delta[4]
                n = 0
                cur_delta = np.zeros(5)
        np.save('time.npy', times)
        np.save('reward.npy', total_rewards)
        np.save('param.npy', np.append(self.theta, self.eta))
        print()


def main():
    agent = Agent()
    agent.train()


if __name__ == '__main__':
    main()
