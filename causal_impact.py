from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

DEFAULT_ARGS = {
    'n_samples': 1000,
    'n_seasons': 7,
}


class CausalImpact:
    """
    Causal inference through counterfactual predictions using a Bayesian structural time-series model.
    """

    def __init__(self, data, inter_date, model_args=None):
        """Main constructor.

        :param pandas.DataFrame data: input data. Must contain at least 2 columns, one being named 'y'.
            See the README for more details.
        :param object inter_date: date of intervention. Must be of same type of the data index elements.
            This should usually be int of datetime.date
        :param {str: object} model_args: parameters of the model
            > n_samples: number of samples in the MCMC sampling

        """
        self.data = data
        self.inter_date = inter_date
        self.model = None
        self.fit = None
        self.model_args = self._check_model_args(model_args)

    def run(self):
        """Fit the BSTS model to the data.
        """
        self.model = UnobservedComponents(
            self.data.loc[:self.inter_date - 1, self._obs_col()].values,
            exog=self.data.loc[:self.inter_date - 1, self._reg_cols()].values,
            level='local linear trend',
            seasonal=self.model_args['n_seasons'],
        )
        self.fit = self.model.fit(
            maxiter=self.model_args['n_samples'],
        )

    def _check_model_args(self, model_args):
        """Check input arguments, and add missing ones if needed.

        :return: the valid dict of arguments
        :rtype: {str: object}
        """
        if model_args is None:
            model_args = {}

        for key, val in DEFAULT_ARGS.iteritems():
            if key not in model_args:
                model_args[key] = val

        return model_args

    def _obs_col(self):
        """Get name of column to be modeled in input data.

        :return: column name
        :rtype: str
        """
        return 'y'

    def _reg_cols(self):
        """Get names of columns used in the regression component of the model.

        :return: the column names
        :rtype: pandas.indexes.base.Index
        """
        return self.data.columns.difference([self._obs_col()])

    def plot_components(self):
        """Plot the estimated components of the model.
        """
        self.fit.plot_components(figsize=(15, 9), legend_loc='lower right')
        plt.show()

    def plot(self):
        """Produce final impact plots.
        """
        min_t = 2 if self.model_args['n_seasons'] is None else self.model_args['n_seasons'] + 1
        # Data model before date of intervention - allows to evaluate quality of fit
        pred = self.fit.get_prediction()
        pre_model = pred.predicted_mean
        pre_lower = pred.conf_int()['lower y'].values
        pre_upper = pred.conf_int()['upper y'].values
        pre_model[:min_t] = np.nan
        pre_lower[:min_t] = np.nan
        pre_upper[:min_t] = np.nan
        # Best prediction of y without any intervention
        post_pred = self.fit.get_forecast(
            steps=self.data.shape[0] - self.inter_date,
            exog=self.data.loc[self.inter_date:, self._reg_cols()]
        )
        post_model = post_pred.predicted_mean
        post_lower = post_pred.conf_int()['lower y'].values
        post_upper = post_pred.conf_int()['upper y'].values

        plt.figure(figsize=(15, 12))

        # Observation and regression components
        ax1 = plt.subplot(3, 1, 1)
        for col in self._reg_cols():
            plt.plot(self.data[col], label=col)
        plt.plot(np.concatenate([pre_model, post_model]), 'r--', linewidth=2, label='model')
        plt.plot(self.data[self._obs_col()], 'k', linewidth=2, label=self._obs_col())
        plt.axvline(self.inter_date, c='k', linestyle='--')
        plt.fill_between(
            self.data.loc[:self.inter_date - 1].index,
            pre_lower,
            pre_upper,
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.fill_between(
            self.data.loc[self.inter_date:].index,
            post_lower,
            post_upper,
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.axis([self.data.index[0], self.data.index[-1], None, None])
        plt.legend(loc='upper left')
        plt.title('Observation vs prediction')

        # Pointwise difference
        ax2 = plt.subplot(312, sharex=ax1)
        plt.plot(self.data[self._obs_col()] - np.concatenate([pre_model, post_model]), 'r--', linewidth=2)
        plt.plot(self.data.index, np.zeros(self.data.shape[0]), 'g-', linewidth=2)
        plt.axvline(self.inter_date, c='k', linestyle='--')
        plt.fill_between(
            self.data.loc[:self.inter_date - 1].index,
            self.data.loc[:self.inter_date - 1, self._obs_col()] - pre_lower,
            self.data.loc[:self.inter_date - 1, self._obs_col()] - pre_upper,
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.fill_between(
            self.data.loc[self.inter_date:].index,
            self.data.loc[self.inter_date:, self._obs_col()] - post_lower,
            self.data.loc[self.inter_date:, self._obs_col()] - post_upper,
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.axis([self.data.index[0], self.data.index[-1], None, None])
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.title('Difference')

        # Cumulative impact
        plt.subplot(313, sharex=ax1)
        plt.plot(
            self.data.loc[self.inter_date:].index,
            (self.data.loc[self.inter_date:, self._obs_col()] - post_model).cumsum(),
            'r--', linewidth=2,
        )
        plt.plot(self.data.index, np.zeros(self.data.shape[0]), 'g-', linewidth=2)
        plt.axvline(self.inter_date, c='k', linestyle='--')
        plt.fill_between(
            self.data.loc[self.inter_date:].index,
            (self.data.loc[self.inter_date:, self._obs_col()] - post_lower).cumsum(),
            (self.data.loc[self.inter_date:, self._obs_col()] - post_upper).cumsum(),
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.axis([self.data.index[0], self.data.index[-1], None, None])
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.title('Cumulative Impact')
        plt.xlabel('$T$')
        plt.show()

        print('Note: the first {} observations are not shown, due to approximate diffuse initialization'.format(min_t))
