# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

DEFAULT_ARGS = {
    'max_iter': 1000,
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
            > max_iter: number of samples in the MCMC sampling
            > n_seasons: number of seasons in the seasonal component of the BSTS model

        """
        # Publicly exposed attributes
        self.data = None            # Input data, with a reset index
        self.data_index = None      # Data initial index
        self.data_inter = None      # Data intervention date, relative to the reset index
        self.model_args = None      # BSTS model arguments
        self.result = None          #
        # Private attributes for modeling purposes only
        self._model = None          # statsmodels BSTS model
        self._fit = None            # statsmodels BSTS fitted model
        # Checking input arguments
        self._check_input(data, inter_date)
        self._check_model_args(data, model_args)

    def _check_input(self, data, inter_date):
        """Check input data.

        :param pandas.DataFrame data: input data. Must contain at least 2 columns, one being named 'y'.
            See the README for more details.
        :param object inter_date: date of intervention. Must be of same type of the data index elements.
            This should usually be int of datetime.date
        """
        self.data_index = data.index
        self.data = data.reset_index(drop=True)
        try:
            self.data_inter = self.data_index.tolist().index(inter_date)
        except ValueError:
            raise ValueError('Input intervention date could not be found in data index.')
        self.result = data.reset_index(drop=False)

    def _check_model_args(self, data, model_args):
        """Check input arguments, and add missing ones if needed.

        :return: the valid dict of arguments
        :rtype: {str: object}
        """
        if model_args is None:
            model_args = {}

        for key, val in DEFAULT_ARGS.items():
            if key not in model_args:
                model_args[key] = val

        if self.data_inter < model_args['n_seasons']:
            raise ValueError('Training data contains more samples than number of seasons in BSTS model.')

        self.model_args = model_args

    def run(self, return_df=False):
        """Fit the BSTS model to the data.
        """
        self._model = UnobservedComponents(
            self.data.loc[:self.data_inter - 1, self._obs_col()].values,
            exog=self.data.loc[:self.data_inter - 1, self._reg_cols()].values,
            level='local linear trend',
            seasonal=self.model_args['n_seasons'],
        )
        self._fit = self._model.fit(
            maxiter=self.model_args['max_iter'],
        )
        self._get_estimates()
        self._get_difference_estimates()
        self._get_cumulative_estimates()

        if return_df:
            return self.result

    def _get_estimates(self):
        """Extracting model estimate (before and after intervention) as well as 95% confidence interval.
        """
        lpred = self._fit.get_prediction()   # Left: model before date of intervention (allows to evaluate fit quality)
        rpred = self._fit.get_forecast(      # Right: best prediction of y without any intervention
            steps=self.data.shape[0] - self.data_inter,
            exog=self.data.loc[self.data_inter:, self._reg_cols()]
        )
        # Model prediction
        self.result = self.result.assign(pred=np.concatenate([lpred.predicted_mean, rpred.predicted_mean]))

        # 95% confidence interval
        lower_conf_ints = []
        upper_conf_ints = []
        for pred in [lpred, rpred]:
            conf_int = pred.conf_int()
            if isinstance(conf_int, np.ndarray):    # As of 0.9.0, statsmodels returns a np.ndarray here
                lower_conf_ints.append(conf_int[:, 0])
                upper_conf_ints.append(conf_int[:, 1])
            else:                                   # instead of a dataframe with "lower y" and "upper y" columns
                lower_conf_ints.append(conf_int.loc[:, 'lower y'].values)
                upper_conf_ints.append(conf_int.loc[:, 'upper y'].values)

        self.result = self.result.assign(pred_conf_int_lower=np.concatenate(lower_conf_ints))
        self.result = self.result.assign(pred_conf_int_upper=np.concatenate(upper_conf_ints))

    def _get_difference_estimates(self):
        """Extracting the difference between the model prediction and the actuals, as well as the related 95%
        confidence interval.
        """
        # Difference between actuals and model
        self.result = self.result.assign(pred_diff=self.data[self._obs_col()].values - self.result['pred'])
        # Confidence interval of the difference
        self.result = self.result.assign(
            pred_diff_conf_int_lower=self.data[self._obs_col()] - self.result['pred_conf_int_upper']
        )
        self.result = self.result.assign(
            pred_diff_conf_int_upper=self.data[self._obs_col()] - self.result['pred_conf_int_lower']
        )

    def _get_cumulative_estimates(self):
        """Extracting estimate of the cumulative impact of the intervention, and its 95% confidence interval.
        """
        # Cumulative sum of modeled impact
        self.result = self.result.assign(cum_impact=0)
        self.result.loc[self.data_inter:, 'cum_impact'] = (
            self.data[self._obs_col()] - self.result['pred']
        ).loc[self.data_inter:].cumsum()

        # Confidence interval of the cumulative sum
        radius_cumsum = np.sqrt(
            ((self.result['pred'] - self.result['pred_conf_int_lower']).loc[self.data_inter:] ** 2).cumsum()
        )
        self.result = self.result.assign(cum_impact_conf_int_lower=0, cum_impact_conf_int_upper=0)
        self.result.loc[self.data_inter:, 'cum_impact_conf_int_lower'] = \
            self.result['cum_impact'].loc[self.data_inter:] - radius_cumsum
        self.result.loc[self.data_inter:, 'cum_impact_conf_int_upper'] = \
            self.result['cum_impact'].loc[self.data_inter:] + radius_cumsum

    def _obs_col(self):
        """Get name of column to be modeled in input data.

        :return: column name
        :rtype: str
        """
        return 'y'

    def _reg_cols(self, small_multiples = False):
        """Get names of columns used in the regression component of the model.

        :return: the column names
        :rtype: pandas.indexes.base.Index
        """
        return self.data.columns.difference([self._obs_col()])

    def plot_components(self):
        """Plot the estimated components of the model.
        """
        self._fit.plot_components(figsize=(15, 9), legend_loc='lower right')
        plt.show()

    def plot(self, small_multiples = False):
        """Produce final impact plots.
        Note: the first few observations are not shown due to approximate diffuse initialization.
        """
        min_t = 2 if self.model_args['n_seasons'] is None else self.model_args['n_seasons'] + 1

        # If we are going to render small multiples for the first row, we want a thinner line
        line_width = .5 if small_multiples else 2

        plt.figure(figsize=(15, 12))
        
        # The overall grid of three rows, one column
        overall = gs.GridSpec(3,1)

        # Observation and regression components
        ax1 = plt.subplot(overall[0])
        main_x_axis_plot = None

        # We will optionally create small multiples - 1 row and n + 1 across (n = length of _reg_cols() plus y & prediction)
        multiple_plot_count = len(self._reg_cols()) + 1
        multiples = gs.GridSpecFromSubplotSpec(1, multiple_plot_count, subplot_spec=overall[0])
        plot_counter = 0
        for col in self._reg_cols():
            if(small_multiples):
                small = plt.subplot(multiples[plot_counter], sharex=main_x_axis_plot)
                if (main_x_axis_plot is None):
                    main_x_axis_plot = small
                plt.axvline(self.data_inter, c='k', linestyle='--')
                plt.title(col)
            plt.plot(self.data[col], label=col, linewidth = line_width)
            plot_counter = plot_counter + 1
        if(small_multiples):
            plt.subplot(multiples[multiple_plot_count - 1], sharex=main_x_axis_plot)
        plt.plot(self.result['pred'].iloc[min_t:], 'r--', linewidth=line_width, label='model')
        plt.plot(self.data[self._obs_col()], 'k', linewidth=line_width, label=self._obs_col())
        plt.axvline(self.data_inter, c='k', linestyle='--')
        plt.fill_between(
            self.data.index[min_t:],
            self.result['pred_conf_int_lower'].iloc[min_t:],
            self.result['pred_conf_int_upper'].iloc[min_t:],
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.legend(loc='upper left')
        plt.title('Observation vs prediction')

        if (main_x_axis_plot is None):
            main_x_axis_plot = ax1

        # Pointwise difference
        ax2 = plt.subplot(overall[1], sharex=main_x_axis_plot)
        plt.plot(self.result['pred_diff'].iloc[min_t:], 'r--', linewidth=2)
        plt.plot(self.data.index, np.zeros(self.data.shape[0]), 'g-', linewidth=2)
        plt.axvline(self.data_inter, c='k', linestyle='--')
        plt.fill_between(
            self.data.index[min_t:],
            self.result['pred_diff_conf_int_lower'].iloc[min_t:],
            self.result['pred_diff_conf_int_upper'].iloc[min_t:],
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.title('Difference')

        # Cumulative impact
        ax3 = plt.subplot(overall[2], sharex=main_x_axis_plot)
        plt.plot(self.data.index, self.result['cum_impact'], 'r--', linewidth=2)
        plt.plot(self.data.index, np.zeros(self.data.shape[0]), 'g-', linewidth=2)
        plt.axvline(self.data_inter, c='k', linestyle='--')
        plt.fill_between(
            self.data.index,
            self.result['cum_impact_conf_int_lower'],
            self.result['cum_impact_conf_int_upper'],
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.axis([self.data.index[0], self.data.index[-1], None, None])
        ax3.set_xticklabels(self.data_index, rotation=45)
        plt.locator_params(axis='x', nbins=min(12, self.data.shape[0]))
        plt.title('Cumulative Impact')
        plt.xlabel('$T$')
        plt.show()
