import pandas as pd

from sklearn.metrics import mean_squared_error


def calc_rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def fps_ml(train_data, test_data, regressor, *args):
    x_train, y_train = train_data.drop(columns=["rating"]), train_data["rating"]
    x_test, y_test = test_data.drop(columns=["rating"]), test_data["rating"]

    rgr = regressor
    if not args:
        rgr.fit(x_train, y_train.values.ravel())
    else:
        if args[0] == "FLAML":
            rgr.fit(x_train, y_train.values.ravel(), task=args[1], time_budget=args[2], metric=args[3])

    pred = rgr.predict(x_test)
    return calc_rmse(y_test, pred)


def dummy_regressor(train_data, test_data, split_info):
    from sklearn.dummy import DummyRegressor

    return fps_ml(train_data, test_data, DummyRegressor())


def rf_regressor(train_data, test_data, split_info):
    from sklearn.ensemble import RandomForestRegressor

    return fps_ml(train_data, test_data, RandomForestRegressor())


def knn_regressor(train_data, test_data, split_info):
    from sklearn.neighbors import KNeighborsRegressor

    return fps_ml(train_data, test_data, KNeighborsRegressor())


def linear_regressor(train_data, test_data, split_info):
    from sklearn.linear_model import LinearRegression

    return fps_ml(train_data, test_data, LinearRegression())


def hist_gradient_boost_regressor(train_data, test_data, split_info):
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingRegressor

    return fps_ml(train_data, test_data, HistGradientBoostingRegressor())


def xgboost_regressor(train_data, test_data, split_info):
    from xgboost import XGBRegressor

    return fps_ml(train_data, test_data, XGBRegressor())


def auto_sklearn_regressor(train_data, test_data, split_info):
    from autosklearn.regression import AutoSklearnRegressor
    from autosklearn.metrics import root_mean_squared_error

    return fps_ml(train_data, test_data,
                  AutoSklearnRegressor(time_left_for_this_task=60, metric=root_mean_squared_error))


def flaml_regressor(train_data, test_data, split_info):
    from flaml import AutoML

    return fps_ml(train_data, test_data, AutoML(), "FLAML", "regression", 60, "rmse")


def h2o_regressor(train_data, test_data, split_info):
    import h2o
    from h2o.automl import H2OAutoML

    x_train, y_train = train_data.drop(columns=["rating"]), train_data["rating"]
    x_test, y_test = test_data.drop(columns=["rating"]), test_data["rating"]

    features = list(x_train)
    label = list(y_train)

    p_x_train = x_train.copy()
    p_x_train[label] = y_train
    train = h2o.H2OFrame(p_x_train)

    rgr = H2OAutoML(max_runtime_secs=60, sort_metric="RMSE")
    rgr.train(x=features, y=label[0], training_frame=train)

    df = h2o.H2OFrame(x_test)
    pred = rgr.predict(df).as_data_frame()

    return calc_rmse(y_test, pred)


def fps_surprise(train_data, test_data, regressor):
    from surprise import Dataset, Reader
    import pandas as pd

    x_train, y_train = train_data.drop(columns=["rating"]), train_data["rating"]
    x_test, y_test = test_data.drop(columns=["rating"]), test_data["rating"]

    label = list(pd.DataFrame(y_train))

    p_x_train = x_train.copy()
    p_x_train[label] = pd.DataFrame(y_train)

    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(p_x_train[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()

    rgr = regressor

    rgr.fit(trainset)

    pred = [
        rgr.predict(getattr(row, "userId"), getattr(row, "movieId"))
        for row in x_test.itertuples()
    ]
    pred = pd.DataFrame(pred)
    pred = pred.rename(
        index=str, columns={"uid": "userId", "iid": "movieId", "est": "prediction"}
    )
    pred = pred.drop(["details", "r_ui"], axis="columns")
    pred = pred['prediction']

    return calc_rmse(y_test, pred)


def svd_surprise(train_data, test_data, split_info):
    from surprise import SVD
    return fps_surprise(train_data, test_data, SVD())


def knn_baseline_surprise(train_data, test_data, split_info):
    from surprise import KNNBaseline
    return fps_surprise(train_data, test_data, KNNBaseline())


def svdpp_surprise(train_data, test_data, split_info):
    from surprise import SVDpp
    return fps_surprise(train_data, test_data, SVDpp())


def fps_lenskit(train_data, test_data, regressor):
    from lenskit.algorithms.basic import Bias, Fallback
    from lenskit.algorithms import Recommender
    import pandas as pd
    x_train, y_train = train_data.drop(columns=["rating"]), train_data["rating"]
    x_test, y_test = test_data.drop(columns=["rating"]), test_data["rating"]

    label = list(pd.DataFrame(y_train))

    p_x_train = x_train.copy()
    p_x_train[label] = pd.DataFrame(y_train)

    p_x_train = p_x_train.rename(columns={"userId": "user", "movieId": "item", "rating": "rating"})

    rgr = regressor

    rgr = Recommender.adapt(rgr)

    base = Bias()
    base = Recommender.adapt(base)

    p_x_train = p_x_train[~p_x_train[["user", "item"]].duplicated()]

    rgr.fit(p_x_train)
    base.fit(p_x_train)

    p_x_test = x_test.copy()

    p_x_test = p_x_test.rename(columns={"userId": "user", "movieId": "item"})
    predictor = Fallback(rgr, base)

    dp_mask = p_x_test[["user", "item"]].duplicated()
    d_p_x_test = p_x_test[~dp_mask]

    pred = predictor.predict(d_p_x_test)

    pred = pred[~pred.index.duplicated()]
    pred = pred.reindex(p_x_test.index)

    for index, row in p_x_test[dp_mask][["user", "item"]].iterrows():
        t = d_p_x_test.index[(d_p_x_test["user"] == row["user"])
                             & (d_p_x_test["item"] == row["item"])].values[0]
        pred_val = pred[t]
        pred[index] = pred_val

    return calc_rmse(y_test, pred)


def user_user(train_data, test_data, split_info):
    from lenskit.algorithms.user_knn import UserUser

    return fps_lenskit(train_data, test_data, UserUser(nnbrs=20))


def item_item(train_data, test_data, split_info):
    from lenskit.algorithms.item_knn import ItemItem

    return fps_lenskit(train_data, test_data, ItemItem(nnbrs=20))


def als_biased_mf(train_data, test_data, split_info):
    from lenskit.algorithms.als import BiasedMF

    return fps_lenskit(train_data, test_data, BiasedMF(features=100))


def bayesian_fm(train_data, test_data, split_info):
    from myfm import MyFMRegressor
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    import scipy.sparse as sps

    x_train, y_train = train_data.drop(columns=["rating"]), train_data["rating"]
    x_test, y_test = test_data.drop(columns=["rating"]), test_data["rating"]

    ohe = OneHotEncoder(handle_unknown="ignore")
    cat_cols = []

    for col in x_train:
        if np.array_equal(x_train[col].values, x_train[col].values.astype(bool)):
            cat_cols.append(col)
    transformed_cols = ohe.fit_transform(x_train.drop(cat_cols, axis=1))
    x_train = sps.hstack([x_train[cat_cols], transformed_cols])

    rgr = MyFMRegressor(rank=10)
    rgr.fit(x_train, y_train.values.ravel(), n_iter=200)

    transformed_cols = ohe.transform(x_test.drop(cat_cols, axis=1))
    x_test = sps.hstack([x_test[cat_cols], transformed_cols])
    pred = rgr.predict(x_test)

    return calc_rmse(y_test, pred)


def prepare_data_libreco(train_data, test_data, split_info, transform_type):
    from LibRecommender.libreco.data import DatasetFeat, DatasetPure

    def rename_and_order(data_in):
        data_in = data_in.rename(columns={"userId": "user", "movieId": "item", "rating": "label"})

        user_col = data_in["user"]
        data_in.pop("user")
        data_in.insert(0, user_col.name, user_col)

        item_col = data_in["item"]
        data_in.pop("item")
        data_in.insert(1, item_col.name, item_col)

        label_col = data_in["label"]
        data_in.pop("label")
        data_in.insert(2, label_col.name, label_col)

        return data_in

    train_data = rename_and_order(train_data)
    test_data = rename_and_order(test_data)

    test_temp = test_data.sample(frac=0.5, random_state=7755720)
    val_data = test_data.drop(test_temp.index)
    test_data = test_temp

    y_test = test_data["label"]
    if transform_type == "PURE":
        train_data = train_data.iloc[:, :3]
        test_data = test_data.iloc[:, :3]
        val_data = val_data.iloc[:, :3]

        val_data = val_data[val_data["user"].isin(train_data["user"])]
        val_data = val_data[val_data["item"].isin(train_data["item"])]

        train_data, data_info = DatasetPure.build_trainset(train_data)
        test_data = DatasetPure.build_testset(test_data)
        val_data = DatasetPure.build_evalset(val_data)
    elif transform_type == "FEAT":
        train_data, data_info = DatasetFeat.build_trainset(
            train_data, split_info["user_cols"], split_info["item_cols"],
            split_info["sparse_cols"], split_info["dense_cols"]
        )
        test_data = DatasetFeat.build_testset(test_data)
        val_data = DatasetFeat.build_evalset(val_data)
    else:
        return None

    return train_data, data_info, test_data, val_data, y_test


def fps_libreco(train_data, test_data, val_data, y_test, data_info, regressor):
    from LibRecommender.libreco.evaluation import computation
    import numpy as np
    import tensorflow as tf2
    tf = tf2.compat.v1
    tf.disable_v2_behavior()

    train_loss, eval_rmse = regressor.fit(train_data, verbose=2, shuffle=True, eval_data=val_data, metrics=["rmse"])
    tf.reset_default_graph()

    pd.DataFrame(train_loss).to_csv("variables/train_loss.csv", header=False, index=False)
    pd.DataFrame(eval_rmse).to_csv("variables/eval_rmse.csv", header=False, index=False)

    epoch = regressor.early_stopping("variables/train_loss.csv", "variables/eval_rmse.csv", window=None,
                                     doc_train=0, doc_eval=0)
    regressor_best = regressor.load_variables('variables', f'model-epoch-{epoch}', data_info)

    pred, _ = computation.compute_preds(model=regressor_best, data=test_data, batch_size=8192)
    pred = np.array(pred)

    tf.reset_default_graph()

    return calc_rmse(y_test, pred)


def svdpp_libreco(train_data, test_data, split_info):
    from LibRecommender.libreco.algorithms import SVDpp
    train_data, data_info, test_data, val_data, y_test = prepare_data_libreco(train_data, test_data, None, "PURE")
    return fps_libreco(train_data, test_data, val_data, y_test, data_info,
                       SVDpp(task="rating", data_info=data_info, embed_size=10, n_epochs=20, lr=.001, reg=.0001,
                             batch_size=256))


def wide_deep(train_data, test_data, split_info):
    from LibRecommender.libreco.algorithms import WideDeep
    train_data, data_info, test_data, val_data, y_test = prepare_data_libreco(train_data, test_data, split_info, "FEAT")
    return fps_libreco(train_data, test_data, val_data, y_test, data_info,
                       WideDeep(task="rating", data_info=data_info, embed_size=64, n_epochs=100,
                                lr={"wide": 0.018467338582854813, "deep": 0.00018065379879874467},
                                lr_decay=False, reg=0.00899680117739739, batch_size=512, num_neg=1, use_bn=True,
                                dropout_rate=0.5, hidden_units="256,128,64", batch_sampling=False,
                                multi_sparse_combiner="sqrtn", seed=42, lower_upper_bound=[1, 5], tf_sess_config=None))


def deep_interest_network(train_data, test_data, split_info):
    from LibRecommender.libreco.algorithms import DIN
    train_data, data_info, test_data, val_data, y_test = prepare_data_libreco(train_data, test_data, split_info, "FEAT")
    return fps_libreco(train_data, test_data, val_data, y_test, data_info,
                       DIN(task="rating", data_info=data_info, embed_size=48, n_epochs=100, lr=0.0028155613535703365,
                           lr_decay=False, reg=0.0012634625195893107, batch_size=192, num_neg=1, use_bn=True,
                           dropout_rate=0.5, hidden_units="256,128,64", recent_num=10, random_num=None,
                           use_tf_attention=False, multi_sparse_combiner="sqrtn", seed=42, lower_upper_bound=[1, 5],
                           tf_sess_config=None))
