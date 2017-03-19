# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymc3 import Model
from pymc3 import sample
from pymc3 import GaussianRandomWalk
from pymc3 import HalfNormal
from pymc3 import Normal


class BSTS:
    """
    Bayesian Structural Time Series.

    Modeling observed time series through a state-space model defined as follows:

        y(t) = m(t) + <beta, x(t)> + e(t)
        m(t) = m(t - 1) + d(t - 1) + u(t)
        d(t) = d(t - 1) + v(t)

        with e, u, v ~ N(0, sigma_e), N(0, sigma_u), N(0, sigma_v)

    """

    def __init__(self):
        """
        """
        self.model = None
        self.trace = None

    def fit(self, features, observed, n_samples):
        """
        """
        with Model() as self.model:
            # Variance of y
            sigma_eps = HalfNormal('sigma_eps', sd=1.)
            # Regression coefs
            alpha = Normal('alpha', mu=0, sd=10)
            beta = Normal('beta', mu=0, sd=10, shape=features.shape[1])
            reg = alpha + sum(beta[i] * features.iloc[:, i] for i in range(features.shape[1]))
            # Trend
            sigma_u = HalfNormal('sigma_u', sd=1.)
            sigma_v = HalfNormal('sigma_v', sd=1.)
            delta = GaussianRandomWalk('delta', sigma_v ** -2, shape=observed.shape[0] - 1)
            trend = GaussianRandomWalk('trend', sigma_u ** -2, mu=delta, shape=observed.shape[0])
            # Observation
            y = Normal('y', mu=trend + reg, sd=sigma_eps, observed=observed)

        with self.model:
            self.trace = sample(
                n_samples,
                progressbar=True,
            )

    def posterior_model(self, features):
        """
        """
        return self.trace['trend'].mean(axis=0) +\
            self.trace['alpha'].mean(axis=0) +\
            (self.trace['beta'].mean(axis=0) * features).sum(axis=1)

    def predict(self, features, noise=True):
        """
        """
        alpha = self.trace['alpha'].mean(axis=0)
        beta = self.trace['beta'].mean(axis=0)
        if noise:
            sigma_eps = self.trace['sigma_eps'].mean(axis=0)
            eps = np.random.normal(loc=0, scale=sigma_eps, size=features.shape[0])
            sigma_u = self.trace['sigma_u'].mean(axis=0)
            u = np.random.normal(loc=0, scale=sigma_u, size=features.shape[0])
            sigma_v = self.trace['sigma_v'].mean(axis=0)
            v = np.random.normal(loc=0, scale=sigma_v, size=features.shape[0])
        else:
            eps = np.zeros(features.shape[0])
            u = np.zeros(features.shape[0])
            v = np.zeros(features.shape[0])

        delta = [self.trace['delta'][:, -1].mean() + v[0]]
        trend = [self.trace['trend'][:, -1].mean() + delta[0] + u[0]]
        for t in range(1, features.shape[0]):
            delta.append(delta[t - 1] + v[t])
            trend.append(trend[t - 1] + delta[t] + u[t])

        return np.array(trend) + alpha + (beta * features).sum(axis=1) + eps

    def predict_trajectories(self, features, n_samples):
        """
        """
        return np.array([self.predict(features) for _ in range(n_samples)])
