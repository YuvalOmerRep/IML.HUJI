from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    samples_noise = np.random.normal(loc=0, scale=noise, size=n_samples)
    X = np.linspace(-1.2, 2, n_samples)

    y_no_noise = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)

    y = y_no_noise + samples_noise

    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion=2/3)

    go.Figure().add_trace(
        go.Scatter(x=X, y=y_no_noise, mode="markers", name="Original data")).add_trace(
        go.Scatter(x=X_train.to_numpy().T[0], y=y_train, mode="markers", name="training data")).add_trace(
        go.Scatter(x=X_test.to_numpy().T[0], y=y_test, mode="markers", name="test data"
                   )).update_layout(title=f"data with {noise} noise").write_image(f"data{noise}.png")

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_loss = []
    validation_loss = []
    degrees = [i for i in range(11)]

    for i in degrees:
        est = PolynomialFitting(i)

        tra, val = cross_validate(est, X_train.to_numpy()[:, 0], y_train.to_numpy(), mean_square_error)
        train_loss.append(tra)
        validation_loss.append(val)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=degrees, y=train_loss, mode="markers", name="train loss")).add_trace(
        go.Scatter(x=degrees, y=validation_loss,
                   mode="markers", name="validation loss")).update_layout(title=f"degree to loss with {noise} noise",
                                                                          xaxis_title="degree of fit",
                                                                          yaxis_title="loss of model"
                                                                          ).write_image(f"Loss_over_degree{noise}.png")

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_eset = PolynomialFitting(np.argmin(validation_loss))

    best_eset.fit(X_train.to_numpy().T[0], y_train.to_numpy())

    print(f"Best value of k is {best_eset.k} for {noise} noise "
          f"which achives test error {np.round(best_eset.loss(X_test.to_numpy().T[0], y_test.to_numpy()), 2)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_samples)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    train_errors_ridge = []
    val_errors_ridge = []

    train_errors_lasso = []
    val_errors_lasso = []

    alpha_x = np.linspace(0, 0.5, n_evaluations)

    for i in alpha_x:
        ridge = RidgeRegression(i)
        tr_er, val_er = cross_validate(ridge, X_train, y_train, mean_square_error)

        train_errors_ridge.append(tr_er)
        val_errors_ridge.append(val_er)

        lasso = Lasso(alpha=i)
        tr_er, val_er = cross_validate(lasso, X_train, y_train, mean_square_error)

        train_errors_lasso.append(tr_er)
        val_errors_lasso.append(val_er)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=alpha_x,
                             y=val_errors_ridge, name="validation error ridge")).add_trace(
        go.Scatter(x=alpha_x, y=train_errors_ridge, name="train error ridge"))

    fig.add_trace(go.Scatter(x=alpha_x, y=val_errors_lasso, name="validation error lasso")).add_trace(
        go.Scatter(x=alpha_x, y=train_errors_lasso, name="train error lasso"))

    fig.write_image("ridge_lasso_alpha_error.png")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge = RidgeRegression(alpha_x[np.argmin(val_errors_ridge)])
    best_lasso = Lasso(alpha=alpha_x[np.argmin(val_errors_lasso)])
    linear = LinearRegression()

    best_ridge.fit(X_train, y_train)
    best_lasso.fit(X_train, y_train)
    linear.fit(X_train, y_train)

    print(f"Best value of lambda in ridge is {best_ridge.lam_} "
          f"which achieved test error {np.round(best_ridge.loss(X_test, y_test), 2)}")

    print(f"Best value of lambda in lasso is {best_lasso.alpha} "
          f"which achieved test error {np.round(mean_square_error(best_lasso.predict(X_test), y_test), 2)}")

    print(f"Linear regression achieved test error {np.round(mean_square_error(linear.predict(X_test), y_test), 2)}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    np.random.seed(0)
    select_polynomial_degree(noise=0)
    np.random.seed(0)
    select_polynomial_degree(n_samples=1500, noise=10)
    np.random.seed(0)
    select_regularization_parameter()
