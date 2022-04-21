import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np


def plot_detailed_summary():
    results = pd.read_csv("./results/paper_results_final.csv")

    feature_sets = ["basic-no-stats", "basic-with-stats", "stripped-no-stats", "stripped-with-stats",
                    "feature-expansion-no-stats", "feature-expansion-with-stats"]
    feature_sets_colors = ["r", "g", "m", "y", "b", "c"]

    algorithms = ["ConstantPredictor_Mean", "lenskit_UserUser", "lenskit_ItemItem", "lenskit_ALSBiasedMF",
                  "Surprise_KNNBaseline", "Surprise_SingularValueDecompositionAlgorithm", "Surprise_SVDpp",
                  "Librecommender_SVDpp", "SciKit_KNeighborsRegressor", "SciKit_LinearRegressor",
                  "SciKit_HistGradientBoostingRegressor", "SciKit_RandomForestRegressor", "XGBoostRegressor",
                  "AutoSKLearn_AutoSklearnRegressor", "H2O_AutoML", "FLAML_Regressor", "Librecommender_WideDeep",
                  "Librecommender_DIN", "myfm_bayesian_factorization_machine"]

    fig = plt.figure(num=0, figsize=(6, 12))
    plt.xlabel("RMSE (lower is better)")
    plt.xlim([0.85, 1.15])
    plt.axvline(0.9, color="k", lw=0.2)
    plt.axvline(0.95, color="k", lw=0.2)
    plt.axvline(1.0, color="k", lw=0.2)
    plt.axvline(1.05, color="k", lw=0.2)
    plt.axvline(1.1, color="k", lw=0.2)
    plt.yticks([])
    legend_patches = []
    for extension_idx, feature_set in enumerate(feature_sets):
        l_patch = mpatches.Patch(color=feature_sets_colors[extension_idx], label=feature_set)
        legend_patches.append(l_patch)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True,
               handles=legend_patches)

    total_pos = 0
    for algo_type in algorithms:
        results_by_algo_type = results[results["algorithm"] == algo_type]
        results_by_algo_type = results_by_algo_type.sort_values(by=["rmse"], ascending=False)
        num_elem = len(results_by_algo_type)
        x = np.arange(start=total_pos, stop=total_pos + num_elem)
        total_pos += (num_elem + 2)
        if not results_by_algo_type.empty:
            data_type_colors = []
            for data_type in results_by_algo_type["feature_set"].values:
                for data_idx, data_set_extension_type in enumerate(feature_sets):
                    if data_set_extension_type in data_type:
                        data_type_colors.append(feature_sets_colors[data_idx])
            plt.barh(x, width=results_by_algo_type["rmse"], color=data_type_colors, height=0.8)

    text_y_pos = [0.94 - (i * 0.06625) for i in range(11)]
    alpha = 0.5
    plt.text(0.2, text_y_pos[0], "Bayesian Factorization Machine", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lime', alpha=alpha))
    plt.text(0.3, text_y_pos[1], "Deep Interest Network", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lime', alpha=alpha))
    plt.text(0.3, text_y_pos[2], "Wide & Deep", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lime', alpha=alpha))
    plt.text(0.3, text_y_pos[3], "FLAML", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='coral', alpha=alpha))
    plt.text(0.3, text_y_pos[4], "H2O", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='coral', alpha=alpha))
    plt.text(0.35, text_y_pos[5], "Auto-Sklearn", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='coral', alpha=alpha))
    plt.text(0.35, text_y_pos[6], "XGBoost", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='orchid', alpha=alpha))
    plt.text(0.45, text_y_pos[7], "Random Forest", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='orchid', alpha=alpha))
    plt.text(0.45, text_y_pos[8], "Histogram Gradient Boosting", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='orchid', alpha=alpha))
    plt.text(0.75, text_y_pos[9], "Linear Regression", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='orchid', alpha=alpha))
    plt.text(0.65, text_y_pos[10], "K Nearest-Neighbors", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='orchid', alpha=alpha))

    text_y_pos = [0.2275 - (i * 0.0245) for i in range(9)]
    plt.text(0.26, text_y_pos[0], "SVDpp (LibRecommender)", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='dodgerblue', alpha=alpha))
    plt.text(0.26, text_y_pos[1], "SVDpp (Surprise)", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='dodgerblue', alpha=alpha))
    plt.text(0.32, text_y_pos[2], "SVD", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='dodgerblue', alpha=alpha))
    plt.text(0.32, text_y_pos[3], "K Nearest Neighbors Baseline", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='dodgerblue', alpha=alpha))
    plt.text(0.26, text_y_pos[4], "Alternating Least Squares Biased", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='dodgerblue', alpha=alpha))
    plt.text(0.26, text_y_pos[5], "ItemItem", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='dodgerblue', alpha=alpha))
    plt.text(0.32, text_y_pos[6], "UserUser", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='dodgerblue', alpha=alpha))
    plt.text(0.75, text_y_pos[8], "Mean Predictor", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='orchid', alpha=alpha))

    plt.text(0.05, 0.99, "RecSys Models", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lime', alpha=alpha))
    plt.text(0.265, 0.99, "AutoML", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='coral', alpha=alpha))
    plt.text(0.4, 0.99, "Machine Learning", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='orchid', alpha=alpha))
    plt.text(0.65, 0.99, "RecSys Matrix Factorization", transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='dodgerblue', alpha=alpha))

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=1, wspace=0.05)

    plt.savefig("./plotting/plot_detailed_summary.png", bbox_inches='tight')
    return


def plot_feature_importance(forest_importances, sorted_std_agg):
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=sorted_std_agg, ax=ax, color="royalblue")
    ax.set_ylabel("Gini importance in percent")
    fig.tight_layout()
    plt.savefig("./plotting/plot_feature_importance.png", bbox_inches='tight')
    return


def plot_rmse_change():
    plt.savefig("./plotting/plot_feature_importance.png", bbox_inches='tight')
    return
