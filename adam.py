import numpy as np


class Adam():
    def __init__(self, f, alpha=0.001, beta1=0.9, beta2=0.999, eps=10**(-8)):
        self.m_prev = np.zeros(6*f)
        self.v_prev = np.zeros(6*f)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def step(self, g, t):
        self.m_prev = self.beta1 * self.m_prev + (1 - self.beta1) * g
        self.v_prev = self.beta2 * self.v_prev + (1 - self.beta2) * np.multiply(g, g)
        m = self.m_prev * (1 / self.beta1 ** t)
        v = self.v_prev * (1 / self.beta2 ** t)
        return self.alpha * np.divide(m, np.power(v, 0.5) + self.eps)