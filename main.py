from preprocessing import process_all_data
from reader import read_preprocessed_data
from features import feature_importance, feature_selection
from hpo import return_best_hp
from evaluation import *
from scoring import score_folds, save_scores

preprocess_data = False
get_feature_importance = False
select_features = False
perform_hpo = True
eval_and_score_all = False


def main():
    if preprocess_data:
        process_all_data()
    datasets = read_preprocessed_data()
    if get_feature_importance:
        feature_importance(datasets["feature-expansion-with-stats"][1]["train"])
    if select_features:
        data_sel = feature_selection(datasets["feature-expansion-with-stats"][1]["train"], 8)
    if perform_hpo:
        best_config = return_best_hp(datasets["feature-expansion-with-stats"][1], "WideDeep")
    if eval_and_score_all:
        scores = score_folds(datasets, [wide_deep])
        save_scores(scores)
    return


if __name__ == '__main__':
    main()
