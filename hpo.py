import os
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    OrdinalHyperparameter
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from LibRecommender.libreco.data import DatasetFeat
from LibRecommender.libreco.algorithms import WideDeep
from LibRecommender.libreco.evaluation import computation
import numpy as np
import pandas as pd
import tensorflow as tf2
import time
from evaluation import calc_rmse

tf = tf2.compat.v1
tf.disable_v2_behavior()


def return_best_hp(dataset, algo):
    if algo == "WideDeep":
        def eval_wide_deep(config):
            start = time.time()

            train_data = dataset["train"]
            test_data = dataset["test"]
            split_info = dataset["split"]

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
            train_data, data_info = DatasetFeat.build_trainset(
                train_data, split_info["user_cols"], split_info["item_cols"],
                split_info["sparse_cols"], split_info["dense_cols"]
            )
            test_data = DatasetFeat.build_testset(test_data)
            val_data = DatasetFeat.build_evalset(val_data)

            regressor = WideDeep(task="rating", data_info=data_info, embed_size=config["embed"], n_epochs=100,
                                 lr={"wide": config["wide_lr"], "deep": config["deep_lr"]},
                                 lr_decay=False, reg=config["reg"], batch_size=config["batch_size"], num_neg=1,
                                 use_bn=True, dropout_rate=0.5, hidden_units="256,128,64", batch_sampling=False,
                                 multi_sparse_combiner="sqrtn", seed=42, lower_upper_bound=[1, 5], tf_sess_config=None)

            train_loss, eval_rmse = regressor.fit(train_data, verbose=2, shuffle=True, eval_data=val_data,
                                                  metrics=["rmse"])
            tf.reset_default_graph()

            pd.DataFrame(train_loss).to_csv("variables/train_loss.csv", header=False, index=False)
            pd.DataFrame(eval_rmse).to_csv("variables/eval_rmse.csv", header=False, index=False)

            epoch = WideDeep.early_stopping("variables/train_loss.csv", "variables/eval_rmse.csv", window=None,
                                            doc_train=0, doc_eval=0)
            regressor_best = WideDeep.load_variables('variables', f'model-epoch-{epoch}', data_info)

            pred, _ = computation.compute_preds(model=regressor_best, data=test_data, batch_size=8192)
            pred = np.array(pred)

            tf.reset_default_graph()

            res_rmse = calc_rmse(y_test, pred)
            print(f"SMAC results. RMSE: {res_rmse}. Config: {config}")
            end = time.time()
            print(f'Network {algo} evaluated in {(end - start):.3f} seconds.')

            variables_dir = "variables/"
            for file in os.listdir(variables_dir):
                if file.endswith(".json") or file.endswith(".csv") or file.endswith(".npz"):
                    os.remove(os.path.join(variables_dir, file))

            return res_rmse

        configspace = ConfigurationSpace()
        configspace.add_hyperparameter(UniformFloatHyperparameter("wide_lr", 0.0001, 0.01))
        configspace.add_hyperparameter(UniformFloatHyperparameter("deep_lr", 0.00001, 0.001))
        configspace.add_hyperparameter(UniformFloatHyperparameter("reg", 0.00001, 0.001))
        configspace.add_hyperparameter(OrdinalHyperparameter("embed", [8, 16, 24, 32]))
        configspace.add_hyperparameter(OrdinalHyperparameter("batch_size", [64, 128, 245]))

        scenario = Scenario({
            "run_obj": "quality",  # Optimize quality (alternatively runtime)
            "runcount-limit": 30,  # Max number of function evaluations (the more the better)
            "cs": configspace,
        })

        smac = SMAC4HPO(scenario=scenario, tae_runner=eval_wide_deep)
        best_config = smac.optimize()
        print(f"SMAC best config: {best_config._values}")
        return best_config

    elif algo == "DIN":
        def eval_din(config):
            start = time.time()

            train_data = dataset["train"]
            test_data = dataset["test"]
            split_info = dataset["split"]

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
            train_data, data_info = DatasetFeat.build_trainset(
                train_data, split_info["user_cols"], split_info["item_cols"],
                split_info["sparse_cols"], split_info["dense_cols"]
            )
            test_data = DatasetFeat.build_testset(test_data)
            val_data = DatasetFeat.build_evalset(val_data)

            regressor = DIN(task="rating", data_info=data_info, embed_size=config["embed"], n_epochs=100,
                            lr=config["lr"], lr_decay=False, reg=config["reg"], batch_size=config["batch_size"],
                            num_neg=1, use_bn=True, dropout_rate=0.5, hidden_units="256,128,64",
                            multi_sparse_combiner="sqrtn", seed=42, lower_upper_bound=[1, 5], tf_sess_config=None)

            train_loss, eval_rmse = regressor.fit(train_data, verbose=2, shuffle=True, eval_data=val_data,
                                                  metrics=["rmse"])
            tf.reset_default_graph()

            pd.DataFrame(train_loss).to_csv("variables/train_loss.csv", header=False, index=False)
            pd.DataFrame(eval_rmse).to_csv("variables/eval_rmse.csv", header=False, index=False)

            epoch = DIN.early_stopping("variables/train_loss.csv", "variables/eval_rmse.csv", window=None,
                                       doc_train=0, doc_eval=0)
            regressor_best = DIN.load_variables('variables', f'model-epoch-{epoch}', data_info)

            pred, _ = computation.compute_preds(model=regressor_best, data=test_data, batch_size=8192)
            pred = np.array(pred)

            tf.reset_default_graph()

            res_rmse = calc_rmse(y_test, pred)
            print(f"SMAC results. RMSE: {res_rmse}. Config: {config}")
            end = time.time()
            print(f'Network {algo} evaluated in {(end - start):.3f} seconds.')

            variables_dir = "variables/"
            for file in os.listdir(variables_dir):
                if file.endswith(".json") or file.endswith(".csv") or file.endswith(".npz"):
                    os.remove(os.path.join(variables_dir, file))

            return res_rmse

        configspace = ConfigurationSpace()
        configspace.add_hyperparameter(UniformFloatHyperparameter("lr", 0.00001, 0.1))
        configspace.add_hyperparameter(UniformFloatHyperparameter("reg", 0.000001, 0.01))
        configspace.add_hyperparameter(OrdinalHyperparameter("embed", [8, 24, 32, 48, 64, 96]))
        configspace.add_hyperparameter(OrdinalHyperparameter("batch_size", [128, 192, 256, 384, 512]))

        scenario = Scenario({
            "run_obj": "quality",  # Optimize quality (alternatively runtime)
            "runcount-limit": 100,  # Max number of function evaluations (the more the better)
            "cs": configspace,
        })

        smac = SMAC4HPO(scenario=scenario, tae_runner=eval_din)
        best_config = smac.optimize()
        print(f"SMAC best config: {best_config._values}")
        return best_config
