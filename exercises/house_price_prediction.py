from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    full_data = pd.read_csv(filename).dropna().drop_duplicates()

    full_data.drop(full_data[full_data.price < 1].index, inplace=True)

    full_data["yr_built_adjusted"] = full_data["yr_built"] - full_data["yr_built"].min()

    full_data["yr_renovated_adjusted"] = (full_data["yr_renovated"] - full_data["yr_built"].min())

    full_data["yr_renovated_adjusted"] = full_data["yr_renovated_adjusted"].mask(
        full_data["yr_renovated"] < full_data["yr_built"], full_data["yr_built_adjusted"], axis=0)

    full_data["zipcode"] = full_data["zipcode"].astype(int)

    full_data = pd.get_dummies(full_data, prefix="zipcode_", columns=["zipcode"])

    labels = full_data["price"]

    features = full_data.drop(["yr_built", "yr_renovated", "id", "date", "price"], axis=1)

    return features, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    pearson_correlation_values = np.apply_along_axis(pearson_correlation, 0, X, np.array(y))

    for index, feature in enumerate(X.keys()):
        go.Figure().add_trace(
            go.Scatter(x=X[feature], y=y,
                       mode="markers")).update_layout(title={
            "text": f"{feature} with Pearson Correlation "
                    f"{pearson_correlation_values[index]}"}).write_image(f"{output_path}/graphs/{feature}.png")


def pearson_correlation(X: np.array, y: np.array) -> float:
    return (np.cov(X, y) / (np.std(X) * np.std(y)))[0][1]


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    house_features, house_prices = load_data(
        "../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(house_features, house_prices)

    # Question 3 - Split samples into training- and testing sets.
    x_train, y_train, x_test, y_test = split_train_test(house_features, house_prices)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    regressor = LinearRegression()
    regressor.fit(np.array(x_train), np.array(y_train))
    print(regressor.loss(np.array(x_test), np.array(y_test)))

    results = []
    for i in range(10, 101):
        loss_regressor = LinearRegression()
        curr_x_train = x_train.sample(frac=i/100)
        loss_regressor.fit(np.array(curr_x_train), np.array(y_train[curr_x_train.index]))
        results.append(loss_regressor.loss(np.array(x_test), np.array(y_test)))
    print(results)
    px.line(x=[i for i in range(10, 101)], y=results).write_image(f"./graphs/loss.png")
