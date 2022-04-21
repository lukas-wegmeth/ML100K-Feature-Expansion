from preprocessing import process_all_data
from reader import read_preprocessed_data
from features import feature_importance, feature_selection
from hpo import return_best_hp
from evaluation import dummy_regressor, rf_regressor, knn_regressor, linear_regressor, hist_gradient_boost_regressor, \
    xgboost_regressor, auto_sklearn_regressor, flaml_regressor, h2o_regressor, svd_surprise, knn_baseline_surprise, \
    svdpp_surprise, user_user, item_item, als_biased_mf, bayesian_fm, svdpp_libreco, wide_deep, deep_interest_network
from scoring import score_folds, save_scores
from plotting import plot_detailed_summary, plot_feature_importance, plot_rmse_change

preprocess_data = False
calc_feature_importance = False
select_features = False
perform_hpo = False
eval_and_score = False
plot_results = True


def main():
    if preprocess_data:
        process_all_data()
    if preprocess_data or calc_feature_importance or select_features or perform_hpo or eval_and_score:
        datasets = read_preprocessed_data()
    if calc_feature_importance:
        importance, std_agg = feature_importance(datasets["feature-expansion-with-stats"][1]["train"])
    if select_features:
        data_sel = feature_selection(datasets["feature-expansion-with-stats"][1]["train"], 8)
    if perform_hpo:
        best_config = return_best_hp(datasets["feature-expansion-with-stats"][1], "DIN")
    if eval_and_score:
        # datasets = {"feature-expansion-with-stats": datasets["feature-expansion-with-stats"]}
        scores = score_folds(datasets, [dummy_regressor])
        save_scores(scores)
    if plot_results:
        plot_detailed_summary()
        if calc_feature_importance:
            plot_feature_importance(importance, std_agg)
        plot_rmse_change()
    return


if __name__ == '__main__':
    main()
