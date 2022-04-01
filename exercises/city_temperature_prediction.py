import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv(filename, parse_dates=["Date"]).drop_duplicates()

    full_data = full_data.drop(full_data[full_data["Temp"] < -20].index)

    full_data["DayOfYear"] = full_data["Date"].apply(lambda x: x.timetuple().tm_yday - 1)

    data = full_data[["Country",
                      "City",
                      "Date",
                      "Year",
                      "Month",
                      "Day",
                      "DayOfYear",
                      "Temp"]]

    return data


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = data[["Temp", "DayOfYear", "Year", "Month"]][data["Country"] == "Israel"]
    israel_data["Year"] = israel_data["Year"].astype(str)

    px.scatter(israel_data, x="DayOfYear", y="Temp", color="Year").write_image("./graphs/DaysToTemp.png")

    px.bar(israel_data[["Month", "Temp"]].groupby("Month").agg("std")).write_image("./graphs/MonthStd.png")

    # Question 3 - Exploring differences between countries
    # raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    # raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()
