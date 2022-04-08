import pandas as pd
import json
import os


def read_preprocessed_data():
    base_folder = 'ml-100k-processed/'
    datasets = {}
    for filename in os.listdir(base_folder):
        file_split = filename.split(";")
        name, set_type, fold = file_split[0], file_split[1], int(file_split[2][:2])
        if set_type == "split":
            with open(f'{base_folder}{filename}') as json_file:
                data_loaded = json.load(json_file)
        else:
            data_loaded = pd.read_csv(f'{base_folder}{filename}')
        if name not in datasets:
            datasets[name] = {}
        if fold not in datasets[name]:
            datasets[name][fold] = {}
        datasets[name][fold][set_type] = data_loaded
    return datasets


def read_raw_data():
    # read data from ml100k download (https://grouplens.org/datasets/movielens/100k/)
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', encoding='iso-8859-1',
                             names=['userId', 'itemId', 'rating', 'timestamp'])

    movies_df = pd.read_csv('ml-100k/u.item', sep='|', encoding="iso-8859-1", header=None)
    movies_df.columns = ['movieId', 'title', 'releaseDate', 'videoReleaseDate', 'imdbUrl', 'unknown', 'action',
                         'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama',
                         'fantasy', 'filmnoir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller',
                         'war', 'western']

    user_df = pd.read_csv('ml-100k/u.user', sep='|', encoding="iso-8859-1", header=None)
    user_df.columns = ['userId', 'age', 'gender', 'occupation', 'zip_code']

    rm_df = pd.merge(movies_df, ratings_df, left_on='movieId', right_on='itemId')
    rm_df = pd.merge(rm_df, user_df, left_on='userId', right_on='userId')

    return rm_df
