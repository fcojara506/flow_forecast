from sklearn.metrics import r2_score, mean_absolute_error
from hydrostats import ens_metrics
from sklearn.metrics import mean_squared_error as mse
import numpy as np
from scipy.stats import rankdata
# Define percentage bias


def pbias(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pbias = np.sum(y_pred - y_true) / np.sum(y_true)
    return pbias * 100


def deterministic_scores(y_true, y_pred):
    """
    It takes in the true values of the target variable and the predicted values of the target variable
    and returns univariate metrics
    
    :param y_train: the actual values of the target variable
    :param y_pred: the predicted values
    :return: The function deterministic_scores returns a dictionary of scores
    """
    score_dict = dict()
    score_dict["rmse_det"] = mse(y_true, y_pred, squared=False)
    score_dict["r2_det"] = r2_score(y_true, y_pred)
    score_dict["pbias"] = pbias(y_true, y_pred)

    return score_dict


def ensemble_scores(y_train, y_ens):
    """
    It takes the true values and ensemble predictions
    and returns ensemble probabilistic metrics
    
    param y_train: observed values
    param y_ens: ensemble predictions
    return: dictionary of scores
    """

    mae_obs = mean_absolute_error(y_train, np.repeat(y_train.mean(), len(y_train)))
    crps_ens = ens_metrics.ens_crps(y_train, y_ens)["crpsMean"]

    score_dict = dict()
    score_dict["mae_obs"] = mae_obs
    score_dict["crps_ens"] = crps_ens
    score_dict["crpss_climatology"] = 1 - crps_ens / mae_obs

    return score_dict


def weights_idv_rank(vector):
    vector = np.array(vector)
    vector = np.array(rankdata(vector))
    vector = vector.reshape((1, -1))
    num = 1 / vector
    return np.array(num)

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def wy_to_year(wy,wym):
    """Take a wateryear(wy) to gregorian year"""
    gregorian_year = wy if wym<10 else wy+1
    return int(gregorian_year)