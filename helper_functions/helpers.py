import pandas as pd
import numpy as np


def lift_score(estimator, X, y):
    """
    Calculate the lift score for each probability decile using a given estimator.

    The function computes the lift score for each decile of predicted probabilities.
    Lift score is a measure used in classification models to evaluate the performance
    of the model. It compares the response rate within a decile of predicted probabilities
    to the average response rate across all observations.

    Parameters:
    -----------
    estimator : object
        A fitted estimator object that has a `predict_proba` method.
    X : array-like, shape (n_samples, n_features)
        The input data used to generate predictions.
    y : array-like, shape (n_samples,)
        The true binary labels (0 or 1).

    Returns:
    --------
    lift_group : pandas.DataFrame
        A DataFrame with the following columns:
        - `decile`: The decile rank (1 to 10, where 1 is the highest predicted probability).
        - `response_rate`: The average response rate (proportion of 1s) within each decile.
        - `count`: The number of samples within each decile.
        - `lift`: The lift score, which is the response rate in the decile divided by the
          overall mean response rate.

    Notes:
    ------
    - The function uses `pd.qcut` to divide the predicted probabilities into deciles.
    - The `predict_proba` method is assumed to return probabilities for the positive class (1).
    """

    lift = pd.DataFrame()
    lift['response'] = y
    lift['predicted_response'] = estimator.predict_proba(X)[:,1]
    lift['decile_rank'] = 10 - pd.qcut(lift['predicted_response'], 10, labels = False, duplicates="drop")
    mean_response_rate = lift['response'].mean()

    lift_group = lift.groupby('deciler_rank').agg({'response':['mean', 'count']})
    lift_group.reset_index(level=0, inplace=True)
    lift_group.columns = ['decile', 'response_rate', 'count']
    lift['lift'] = lift_group['response_rate']/mean_response_rate
    return lift_group

