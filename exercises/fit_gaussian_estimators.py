from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"
pio.renderers.default = "browser"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    fitter = UnivariateGaussian()
    fitter.fit(samples)
    print(f"({fitter.mu_}, {fitter.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    results = []
    for i in range(1, 101):
        fitter2 = UnivariateGaussian()
        results.append(abs(10 - fitter2.fit(samples[:(i * 10)]).mu_))

    fig = px.line(x=[i * 10 for i in range(1, 101)], y=results,
                  labels=dict(x="Sample Size", y="abs distance between estimation and true"))
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    result = fitter.pdf(samples)

    fig2 = px.scatter(x=samples, y=result, labels=dict(x="original sample value", y="PDF value"))
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])

    samples = np.random.multivariate_normal(mu, sigma, 1000)

    mult = MultivariateGaussian()

    mult.fit(samples)

    print(mult.mu_)
    print(mult.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)

    results = []
    for i in f1:
        for j in f3:
            mu1 = np.array([i, 0, j, 0])
            results.append(MultivariateGaussian.log_likelihood(mu1, sigma, samples))

    figure2 = px.density_heatmap(x=f1, y=f3, z=results, histfunc="avg", histnorm="density")
    figure2.show()

    # Question 6 - Maximum likelihood


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
