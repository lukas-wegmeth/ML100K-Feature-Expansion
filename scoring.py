import pandas as pd
import numpy as np
import time


def score_folds(data, algos):
    scores = {}
    for algo in algos:
        algo_name = algo.__name__
        print(f'{"".join(["#" for i in range(10)])} Evaluating {algo_name}.')
        for set_name in data:
            print(f'{"".join(["#" for i in range(5)])} On data set {set_name}.')
            for fold_id, data_parts in data[set_name].items():
                train_data = data_parts["train"]
                test_data = data_parts["test"]
                split_info = data_parts["split"]

                if algo_name not in scores:
                    scores[algo_name] = {}
                if set_name not in scores[algo_name]:
                    scores[algo_name][set_name] = {}

                start = time.time()
                eval_score = algo(train_data, test_data, split_info)
                end = time.time()
                print(f'Fold {fold_id} RMSE: {eval_score:.4f}. Evaluated in {(end - start):.3f} seconds.')

                scores[algo_name][set_name][fold_id] = eval_score

    return scores


def save_scores(scores):
    for algo in scores:
        for data_set in scores[algo]:
            scores_folds = list(scores[algo][data_set].values())
            mean_score = np.array(scores_folds).mean()
            pd.DataFrame(
                {"algo": algo, "data_set": data_set, "scores": [scores_folds], "mean_score": mean_score}).to_csv(
                "results/results.csv", mode='a', header=False, index=False)

    return
