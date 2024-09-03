import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap as shap

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay

__all__ = ['lift_score', 'eval_metrics']

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

    lift_group = lift.groupby('decile_rank').agg({'response':['mean', 'count']})
    lift_group.reset_index(level=0, inplace=True)
    lift_group.columns = ['decile', 'response_rate', 'count']
    lift_group['lift'] = lift_group['response_rate']/mean_response_rate
    return lift_group


def eval_metrics(estimator, X, y, model_name=""):
    """
    Evaluate and display key metrics for a classification model, including a classification report,
    confusion matrix, ROC curve, and lift chart.

    This function evaluates the performance of a classification model using several metrics and visualizations:
    - Prints a classification report summarizing precision, recall, f1-score, and support.
    - Displays a confusion matrix.
    - Displays a ROC curve.
    - Computes and prints a lift score report.
    - Plots a lift chart based on deciles of predicted probabilities.

    Parameters:
    -----------
    estimator : object
        A fitted estimator object that has `predict` and `predict_proba` methods.
    X : array-like, shape (n_samples, n_features)
        The input data used to generate predictions.
    y : array-like, shape (n_samples,)
        The true binary labels (0 or 1).
    model_name : str, optional (default="")
        An optional name for the model, used for labeling and titles.

    Notes:
    ------
    - The function assumes that `classification_report`, `ConfusionMatrixDisplay`, and `RocCurveDisplay`
      are imported from `sklearn.metrics` and `matplotlib.pyplot` is imported as `plt`.
    - The function relies on the `lift_score` function to calculate and display the lift report and lift chart.
    - The lift chart is plotted using `matplotlib` and displays lift across deciles.
    """

    y_pred_test = estimator.predict(X)
    class_report_test = classification_report(y, y_pred_test)
    print("Model metrics Summary: \n%s", str(class_report_test))

    display = ConfusionMatrixDisplay.from_estimator(estimator, X, y)
    _ = display.ax_.set_title("Confusion Matrix")

    display2 = RocCurveDisplay.from_estimator(estimator, X, y)
    _ = display2.ax_.set_title("ROC Curve")

    model_lift = lift_score(estimator, X, y)
    print("Model Lift Report: \n%s", str(model_lift))

    plt.figure(4)
    ax=plt.gca()
    model_lift.plot(x='decile', y="lift", ax=ax, color='r')
    ax.legend(["Lift"])
    plt.title("Lift Chart")
    plt.xlabel('Decile')
    ax.grid('on')
    plt.xlim([1,10])
    return 


def plot_shap_summary_and_importance(model, X, max_display=10, sample_size=10000, random_state=None):
    """
    Plot the SHAP values summary and feature importance based on SHAP values for a random sample.
    
    Parameters:
    - model: A trained XGBoost model (e.g., xgb.Booster or xgb.XGBClassifier).
    - X: The data on which SHAP values will be computed (pandas DataFrame or numpy array).
    - max_display: The maximum number of top features to display in the plots.
    - sample_size: The number of samples to randomly select from X for SHAP analysis.
    - random_state: Seed for random sampling to ensure reproducibility.
    """
    # Sample the data
    if isinstance(X, pd.DataFrame):
        X_sampled = X.sample(n=sample_size, random_state=random_state)
    else:
        X_sampled = X[np.random.RandomState(seed=random_state).choice(X.shape[0], size=sample_size, replace=False)]

    # Ensure SHAP is supported for the model
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sampled)
    
    # Plot the SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sampled, max_display=max_display)
    plt.show()
    
    # Plot the SHAP feature importance (mean absolute value of SHAP values)
    plt.figure(figsize=(10, 6))
    # shap.summary_plot(shap_values, X_sampled, max_display=max_display, plot_type="bar")
    shap.plots.bar(shap_values)
    plt.show()