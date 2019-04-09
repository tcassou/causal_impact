# Python Causal Impact

[![Build Status](https://travis-ci.org/tcassou/causal_impact.svg?branch=master)](https://travis-ci.org/tcassou/causal_impact)

**Work In Progress**

Causal inference using Bayesian structural time-series models.
This package aims at defining a python equivalent of the [R CausalImpact package by Google](https://github.com/google/CausalImpact).
Please refer to [the package](https://github.com/google/CausalImpact) itself, [its documentation](http://google.github.io/CausalImpact/CausalImpact.html) or the [related publication](http://research.google.com/pubs/pub41854.html) (Brodersen et al., Annals of Applied Statistics, 2015) for more information.

## Setup

Simply install from `pip`:
```
pip install causal-impact
```

## Example

Suppose we have a `DataFrame` `data` recording daily measures for three different markets `y`, `x1` and `x2`,  for `t = 0..365`).
The `y` time series in `data` is the one we will be modeling, while other columns (`x1` and `x2` here) will be used as a set of control time series.
```
>>> data
      y       x1      x2
  0   1735.01 1014.44 1005.87
  1   1709.54 1012.63 1008.18
  2   1772.95 1039.04 1024.21
...   ...     ...     ...
```
At `t = date_inter = 280`, a marketing campaing (the *intervention*) is run for market `y`. We want to understand the impact of that campaign on our measure.

```
from causal_impact.causal_impact import CausalImpact

ci = CausalImpact(data, date_inter)
ci.run()
ci.plot()
```

After fitting the model, and estimating what the `y` time series would have been without any intervention, this will typically produce the following plots:
![Impact Plot](https://github.com/tcassou/causal_impact/blob/master/examples/causal_impact.png)

Alternatively, if you would like to render each time series in a separate chart as small multiples, this may be specified in the plot() function
```
ci.plot(small_multiples = True)
```
![Small Multiples Plot](https://github.com/tcassou/causal_impact/blob/master/examples/small_multiples.png)


If you need access to the data behind the plots for further analysis, you can simply use the `ci.result` attribute (`pandas.DataFrame` object). Alternatively, you can also call
```
result = ci.run(return_df=True)
```
and skip the plotting step.

## Issues and improvements
As mentioned above, this package is still being developed. Feel free to contribute through github by sending pull requests or reporting issues.
