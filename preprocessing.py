import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import kurtosis, skew
import json
from reader import read_raw_data
import copy


def process_all_data():
    stripped_name, stripped_data, stripped_info = preprocess_ml_100k_stripped()
    basic_name, basic_data, basic_info = preprocess_ml_100k()
    feature_expansion_name, feature_expansion_data, feature_expansion_info = preprocess_ml_100k_feature_expansion()

    split_and_save_data(stripped_name, stripped_data, stripped_info, False)
    split_and_save_data(stripped_name, stripped_data, stripped_info, True)

    split_and_save_data(basic_name, basic_data, basic_info, False)
    split_and_save_data(basic_name, basic_data, basic_info, True)

    split_and_save_data(feature_expansion_name, feature_expansion_data, feature_expansion_info, False)
    split_and_save_data(feature_expansion_name, feature_expansion_data, feature_expansion_info, True)


def split_and_save_data(name, data, info, with_stats):
    folds_it = KFold(n_splits=5, shuffle=True, random_state=7755720)
    idx = 0

    data = data[:500]

    for train_index, test_index in folds_it.split(data):
        name_fold = name
        info_fold = copy.deepcopy(info)
        idx += 1
        train, test = data.loc[train_index], data.loc[test_index]
        if with_stats:
            user_col = "userId"
            item_col = "movieId"
            rating_col = "rating"

            stat_names = ["Mean", "Median", "Mode", "Min", "Max", "Std", "Kurt", "Skew", "Count"]

            unique_user_ids = train[user_col].unique()
            u_stats = dict()
            for user_id in unique_user_ids:
                u_r = train[train[user_col] == user_id][rating_col]
                u_stats[user_id] = [u_r.mean(), u_r.median(), u_r.mode().values[0], u_r.min(),
                                    u_r.max(), u_r.std(ddof=0), kurtosis(u_r), skew(u_r), u_r.count()]

            unique_item_ids = train[item_col].unique()
            i_stats = dict()
            for item_id in unique_item_ids:
                i_r = train[train[item_col] == item_id][rating_col]
                i_stats[item_id] = [i_r.mean(), i_r.median(), i_r.mode().values[0], i_r.min(),
                                    i_r.max(), i_r.std(ddof=0), kurtosis(i_r), skew(i_r), i_r.count()]

            t_r = train[rating_col]
            g_stats = [t_r.mean(), t_r.median(), t_r.mode().values[0], t_r.min(),
                       t_r.max(), t_r.std(ddof=0), kurtosis(t_r), skew(t_r), 0]

            for stat_idx, stat_name in enumerate(stat_names):
                train[f"u{stat_name}Rating"] = train.apply(
                    lambda row: u_stats[row[user_col]][stat_idx]
                    if u_stats.get(row[user_col]) is not None
                    else g_stats[stat_idx], axis=1)
                test[f"u{stat_name}Rating"] = test.apply(
                    lambda row: u_stats[row[user_col]][stat_idx]
                    if u_stats.get(row[user_col]) is not None
                    else g_stats[stat_idx], axis=1)
                info_fold["user_cols"].append(f"u{stat_name}Rating")
                info_fold["dense_cols"].append(f"u{stat_name}Rating")

            for stat_idx, stat_name in enumerate(stat_names):
                train[f"i{stat_name}Rating"] = train.apply(
                    lambda row: i_stats[row[item_col]][stat_idx]
                    if i_stats.get(row[item_col]) is not None
                    else g_stats[stat_idx], axis=1)
                test[f"i{stat_name}Rating"] = test.apply(
                    lambda row: i_stats[row[item_col]][stat_idx]
                    if i_stats.get(row[item_col]) is not None
                    else g_stats[stat_idx], axis=1)
                info_fold["item_cols"].append(f"i{stat_name}Rating")
                info_fold["dense_cols"].append(f"i{stat_name}Rating")

            name_fold += "-with-stats"
        else:
            name_fold += "-no-stats"

        train.to_csv(f'ml-100k-processed/{name_fold};train;{idx:02}.csv', sep=',', header=True, index=False)
        test.to_csv(f'ml-100k-processed/{name_fold};test;{idx:02}.csv', sep=',', header=True, index=False)
        with open(f'ml-100k-processed/{name_fold};split;{idx:02}.json', 'w', encoding='utf-8') as f:
            json.dump(info_fold, f, ensure_ascii=False, indent=4)


def date_to_timestamp(data, to_encode_columns, prefix=False):
    for col in to_encode_columns:
        df_dates = pd.to_datetime(data[col]).apply(lambda x: int(pd.Timestamp(x).value / 10 ** 9))

        if prefix:
            df_dates.name = "ts_" + col

        data = data.drop([col], axis=1)
        data = pd.concat([data, df_dates], axis=1)
    return data


def preprocess_ml_100k_stripped():
    rm_df = read_raw_data()
    rm_df = rm_df[["userId", "movieId", "rating"]]

    data_info = {"sparse_cols": [], "dense_cols": [], "user_cols": [], "item_cols": []}

    return 'stripped', rm_df, data_info


def preprocess_ml_100k():
    rm_df = read_raw_data()

    dense_cols = ['age']
    sparse_cols = ['unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary',
                   'drama', 'fantasy', 'filmnoir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller',
                   'war', 'western']
    user_cols = ['age']
    item_cols = ['unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary',
                 'drama', 'fantasy', 'filmnoir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller',
                 'war', 'western']

    to_encode_categorical = ['occupation', 'gender']
    for col in to_encode_categorical:
        df_dummies = pd.get_dummies(rm_df[col], prefix=col)
        sparse_cols = sparse_cols + list(df_dummies)
        user_cols = user_cols + list(df_dummies)
        rm_df = pd.concat([rm_df, df_dummies], axis=1)

    rm_df = date_to_timestamp(rm_df, ['releaseDate'], prefix=True)

    to_drop = ['title', 'imdbUrl', 'itemId', 'zip_code', 'videoReleaseDate'] + to_encode_categorical
    rm_df = rm_df.drop(to_drop, axis=1)

    data_info = {"sparse_cols": sparse_cols, "dense_cols": dense_cols, "user_cols": user_cols, "item_cols": item_cols}

    return 'basic', rm_df, data_info


def preprocess_ml_100k_feature_expansion():
    rm_df = read_raw_data()

    dense_cols = []
    sparse_cols = ['unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary',
                   'drama', 'fantasy', 'filmnoir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller',
                   'war', 'western']
    user_cols = []
    item_cols = ['unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary',
                 'drama', 'fantasy', 'filmnoir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller',
                 'war', 'western']

    rm_df = rm_df[rm_df["zip_code"].str.isnumeric()]
    rm_df["zip_code"] = rm_df["zip_code"].astype(np.int64)

    income_df = pd.read_excel(io='ml-100k/income.xlsx')
    dense_cols = dense_cols + ["median_income", "mean_income", "population"]
    user_cols = user_cols + ["median_income", "mean_income", "population"]
    income_df.columns = ["zip_code", "median_income", "mean_income", "population"]
    income_df["median_income"] = income_df["median_income"].apply(lambda inc: round(inc))
    income_df = income_df[income_df["mean_income"] != "."]
    income_df["mean_income"] = income_df["mean_income"].apply(lambda inc: round(inc))
    rm_df = pd.merge(rm_df, income_df, left_on='zip_code', right_on='zip_code')

    rm_df["zip_code"] = rm_df["zip_code"].apply(lambda zipc: int(str(zipc)[0]) if zipc >= 10000 else 0)

    rm_df["age"] = rm_df["age"].apply(lambda age: age // 18)

    to_encode_categorical = ["occupation", "gender", "age", "zip_code"]

    for col in to_encode_categorical:
        df_dummies = pd.get_dummies(rm_df[col], prefix=col)
        sparse_cols = sparse_cols + list(df_dummies)
        user_cols = user_cols + list(df_dummies)
        rm_df = pd.concat([rm_df, df_dummies], axis=1)

    rm_df = date_to_timestamp(rm_df, ['releaseDate'], prefix=True)

    to_drop = ['title', 'imdbUrl', 'itemId', 'videoReleaseDate'] + to_encode_categorical
    rm_df = rm_df.drop(to_drop, axis=1)

    data_info = {"sparse_cols": sparse_cols, "dense_cols": dense_cols, "user_cols": user_cols, "item_cols": item_cols}

    return 'feature-expansion', rm_df, data_info
