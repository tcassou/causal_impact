# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from datetime import date
from datetime import timedelta

import numpy as np
import pandas as pd
from genty import genty
from genty import genty_dataset
from nose.tools import eq_
from nose.tools import ok_
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises

from causal_impact import CausalImpact


@genty
class CausalImpactTest(unittest.TestCase):

    @staticmethod
    def mock_data(start_date, n_dates, inter_index, n_regressors):
        # Generating random regressors
        regressors = [
            np.random.rand() * 100 + 2 * np.random.rand() * np.random.randn(n_dates).cumsum()
            for _ in range(n_regressors)
        ]
        # Output from regressors + noise
        y = sum(2 * np.random.rand() * r for r in regressors) + np.random.randn(n_dates)
        # Adding artificial impact
        t = np.arange(n_dates - inter_index) / (n_dates - inter_index)
        i = np.random.rand() * 10000 * t ** 2 * np.exp(-n_dates / 10 * t)
        y[inter_index:] += i
        # Concatenating into a dataframe
        df = pd.concat(
            [pd.Series(r, name='x{}'.format(i)) for i, r in enumerate(regressors)] + [pd.Series(y, name='y')], axis=1
        )
        df.index = [start_date + timedelta(d) for d in range(n_dates)]
        return df

    def test_init(self):
        data = CausalImpactTest.mock_data(date(2018, 1, 1), 30, 20, 2)
        ci = CausalImpact(data, date(2018, 1, 20), 5)
        assert_array_equal(ci.data, data.reset_index(drop=True))
        assert_array_equal(ci.result, data.reset_index(drop=True))
        eq_(ci.inter_date, date(2018, 1, 20))
        eq_(ci.n_seasons, 5)
        assert_array_equal(ci._input_index, data.index)
        eq_(ci._inter_index, 19)
        ok_(ci._model is None)
        ok_(ci._fit is None)

    @genty_dataset(
        inter_not_found=(date(2018, 3, 1), 7),
        too_many_seasons=(date(2018, 1, 20), 35),
        too_few_seasons=(date(2018, 1, 20), 1),
    )
    def test_init_wrong_input(self, date_inter, n_seasons):
        data = CausalImpactTest.mock_data(date(2018, 1, 1), 30, 20, 2)
        assert_raises(ValueError, CausalImpact, data, date_inter, n_seasons)
