from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    fitter = UnivariateGaussian()
    fitter.fit(samples)
    print(fitter.mu_, fitter.var_)

    # Question 2 - Empirically showing sample mean is consistent
    results = []
    for i in range(1, 101):
        fitter2 = UnivariateGaussian()
        results.append(abs(10 - fitter2.fit(samples[0:(i*10 + 1)]).mu_))

    fig = px.scatter([i*10 for i in range(1, 101)], results)
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
