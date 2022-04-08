from sklearn.metrics import mean_squared_error, mean_absolute_error


def calc_rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def calc_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
