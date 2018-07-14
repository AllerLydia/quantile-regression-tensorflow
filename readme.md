# Deep Quantile Regression Implementations

Based on [strongio/quantile-regression-tensorflow](https://github.com/strongio/quantile-regression-tensorflow), and with the following expansions:

1. Use the example dataset from [the scikit-learn example](http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html).
2. The TensorFlow implementation is mostly the same as in [strongio/quantile-regression-tensorflow](https://github.com/strongio/quantile-regression-tensorflow).
3. Add a LightGBM "quantile" objective example (and a scikit-learn GBM example for comparison) based on this [Github issue](https://github.com/Microsoft/LightGBM/issues/1182).
4. Add a Pytorch implementation.

## Dockerfile
The accompanied *Dockerfile* contains all the dependencies needed to reproduce the results.
