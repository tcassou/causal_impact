from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.bsts import BSTS

DEFAULT_ARGS = {
    'n_samples': 1000,
}


class CausalImpact:
    """
    Causal inference through counterfactual predictions using a Bayesian structural time-series model.
    """

    def __init__(self, data, inter_date, model_args=None):
        """
        """
        self.data = data
        self.inter_date = inter_date
        self.t_step = data.index[1] - data.index[0]
        self.model = None
        self.model_args = self._check_model_args(model_args)

    def run(self):
        """
        """
        self.model = BSTS()
        self.model.fit(
            self.data.loc[:self.inter_date - self.t_step, self._reg_cols()],
            self.data.loc[:self.inter_date - self.t_step, self._obs_col()],
            n_samples=self.model_args['n_samples'],
        )
        return self.model

    def _check_model_args(self, model_args):
        """
        """
        if model_args is None:
            model_args = {}

        for key, val in DEFAULT_ARGS.iteritems():
            if key not in model_args:
                model_args[key] = val

        return model_args

    def _obs_col(self):
        """
        """
        return 'y'

    def _reg_cols(self):
        """
        """
        return self.data.columns.difference([self._obs_col()])

    def plot(self):
        """
        """
        pre_model = self.model.posterior_model(self.data.loc[:self.inter_date - self.t_step, self._reg_cols()])
        pre_var = self.model.trace['sigma_eps'].mean()
        post_pred = self.model.predict(self.data.loc[self.inter_date:, self._reg_cols()], noise=False)
        trajectories = self.model.predict_trajectories(
            self.data.loc[self.inter_date:, self._reg_cols()], self.model_args['n_samples'])
        std_traj = np.std(trajectories, axis=0)

        plt.figure(figsize=(15, 12))
        # Observation and regression components
        ax1 = plt.subplot(3, 1, 1)
        for col in self._reg_cols():
            plt.plot(self.data[col], label=col)
        plt.plot(pd.concat([pre_model, post_pred]), 'r--', linewidth=2, label='model')
        plt.plot(self.data[self._obs_col()], 'k', linewidth=2, label=self._obs_col())
        plt.axvline(self.inter_date, c='k', linestyle='--')
        plt.fill_between(
            self.data.loc[:self.inter_date - self.t_step].index,
            pre_model - 2 * pre_var,
            pre_model + 2 * pre_var,
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.fill_between(
            self.data.loc[self.inter_date:].index,
            post_pred - 1 * std_traj,
            post_pred + 1 * std_traj,
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.axis([self.data.index[0], self.data.index[-1], None, None])
        plt.legend(loc='upper left')
        plt.title('Observation vs prediction')
        # Pointwise difference
        ax2 = plt.subplot(312, sharex=ax1)
        plt.plot(self.data[self._obs_col()] - pd.concat([pre_model, post_pred]), 'r--', linewidth=2)
        plt.plot(self.data.index, np.zeros(self.data.shape[0]), 'g-', linewidth=2)
        plt.axvline(self.inter_date, c='k', linestyle='--')
        plt.fill_between(
            self.data.loc[:self.inter_date - self.t_step].index,
            self.data.loc[:self.inter_date - self.t_step, self._obs_col()] - pre_model - 2 * pre_var,
            self.data.loc[:self.inter_date - self.t_step, self._obs_col()] - pre_model + 2 * pre_var,
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.fill_between(
            self.data.loc[self.inter_date:].index,
            self.data.loc[self.inter_date:, self._obs_col()] - post_pred - 1 * std_traj,
            self.data.loc[self.inter_date:, self._obs_col()] - post_pred + 1 * std_traj,
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.axis([self.data.index[0], self.data.index[-1], None, None])
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.title('Difference')
        # Cumulative impact
        ax3 = plt.subplot(313, sharex=ax1)
        plt.plot(
            self.data.loc[self.inter_date:].index,
            (self.data.loc[self.inter_date:, self._obs_col()] - post_pred).cumsum(),
            'r--', linewidth=2,
        )
        plt.plot(self.data.index, np.zeros(self.data.shape[0]), 'g-', linewidth=2)
        plt.axvline(self.inter_date, c='k', linestyle='--')
        plt.fill_between(
            self.data.loc[self.inter_date:].index,
            (self.data.loc[self.inter_date:, self._obs_col()] - post_pred - 1 * std_traj).cumsum(),
            (self.data.loc[self.inter_date:, self._obs_col()] - post_pred + 1 * std_traj).cumsum(),
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.axis([self.data.index[0], self.data.index[-1], None, None])
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.title('Cumulative Impact')
        plt.xlabel('$T$')
        plt.show()
