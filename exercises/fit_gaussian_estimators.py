from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
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

    px.line(x=[i * 10 for i in range(1, 101)], y=results,
            labels=dict(x="Sample Size", y="abs distance between estimation and true"),
            title="Distance from true expectation to estimated expectation over different sample sizes").show()

    # Question 3 - Plotting Empirical PDF of fitted model
    result = fitter.pdf(samples)

    px.scatter(x=samples, y=result, labels=dict(x="original sample value", y="PDF value"),
               title="PDF values returned by the fitted model over their sampled value").show()


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

    results = np.zeros(40000)
    axis_values = np.zeros([40000, 2])
    count = 0
    for i in f1:
        for j in f3:
            mu1 = np.array([i, 0, j, 0])
            results[count] = MultivariateGaussian.log_likelihood(mu1, sigma, samples)
            axis_values[count] = [i, j]
            count += 1

    px.density_heatmap(x=axis_values[:, 1], y=axis_values[:, 0], z=results, histfunc="avg", histnorm="density",
                       labels=dict(x="values from f3", y="values from f1"),
                       title="Density heat map of log likelihood with relation to "
                             "changing values of indexes in the expectation").show()

    # Question 6 - Maximum likelihood
    maximum = np.amax(results)
    index_of_maximum = np.where(results == maximum)
    axis_indexes = axis_values[index_of_maximum][0]
    f1_value = axis_indexes[0]
    f3_value = axis_indexes[1]

    print(f"The maximum log-likelihood was achieved with {f1_value} for f1\n"
          f"and {f3_value} for f3 with the log-likelihood value of {round(maximum, 3)}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
