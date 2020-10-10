"""Script run at the owner site and the local sites.

At iteration 0, run local_0(): Parses data. At the owner site, saves the parsed 
    files to cache. At each local site, trains a classifier, tests on the 
    training data and sends performance metrics to the remote site. 
At iteration 1, run local_1(): At the owner site, trains an aggregator 
    classifier, tests all local and aggregator classifiers on the testing data 
    and sends the aggregator classifier and performance metrics to the remote 
    site. At local sites, nothing.

Raises:
    Exception: If neither 'phase' nor 'remote_0' is in sys.stdin.
"""
import os
import sys

import copy
import numpy as np
import ujson as json
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from dp_stats.train_predict import (
    train_model,
    predict_linearmodel,
    predict_proba_lr,
    predict_decision_svmhuber,
)
from common_functions import list_recursive
from parsers import fsl_parser


def local_0(args):
    input = args["input"]
    state = args["state"]
    cache_dir = state["cacheDirectory"]
    base_dir = state["baseDirectory"]
    owner = state["owner"] if "owner" in state else "local0"

    (X, y) = fsl_parser(input, base_dir)

    if state["clientId"] == owner:
        np.save(os.path.join(cache_dir, "X.npy"), X)
        np.save(os.path.join(cache_dir, "y.npy"), y)

        cache_dict = {
            "model_local": input["model_local"],
            "model_owner": input["model_owner"],
            "is_private_owner": input["is_private_owner"],
            "perturb_method_owner": input["perturb_method_owner"],
            "lambda_owner": input["lambda_owner"],
            "epsilon_owner": input["epsilon_owner"],
            "huberconst_owner": input["huberconst_owner"],
            "fit_intercept_owner": input["fit_intercept_owner"],
            "intercept_scaling_owner": input["intercept_scaling_owner"],
            "train_split": input["train_split"],
            "shuffle": input["shuffle"],
            "X_filename": "X.npy",
            "y_filename": "y.npy",
        }
        output_dict = {"phase": "local_0"}
    else:
        n_samples_local = X.shape[0]
        # preprocess training data
        if input["fit_intercept_local"]:
            # add synthetic feature
            synthetic = input["intercept_scaling_local"] * np.ones(
                (n_samples_local, 1)
            )
            X_scaled = np.concatenate((X, synthetic), axis=1)
        else:
            X_scaled = copy.deepcopy(X)
        # required by dp: each training sample vector ||x_i|| <= 1
        scale = np.amax(np.linalg.norm(X_scaled, axis=1))
        X_scaled = X_scaled / scale

        # train local model
        w_local = (1 / scale) * train_model(X_scaled, y, input, "local")

        if input["fit_intercept_local"]:
            intercept_local = input["intercept_scaling_local"] * w_local[-1]
            w_local = w_local[:-1]
        else:
            intercept_local = 0.0

        # predict on training data, calculate confusion matrix and accuracy
        y_train_pred = predict_linearmodel(w_local, intercept_local, X)
        cm_train_local = confusion_matrix(y, y_train_pred, normalize=None)
        cm_train_local_normalized = confusion_matrix(
            y, y_train_pred, normalize="true"
        )
        acc_train_local = accuracy_score(y, y_train_pred, normalize=True)
        cache_dict = {}

        output_dict = {
            "w_local": w_local.tolist(),
            "intercept_local": float(intercept_local),
            "cm_train_local": cm_train_local.tolist(),
            "cm_train_local_normalized": cm_train_local_normalized.tolist(),
            "acc_train_local": float(acc_train_local),
            "n_samples_local": n_samples_local,
            "phase": "local_0",
        }

    result_dict = {"output": output_dict, "cache": cache_dict}
    return json.dumps(result_dict)


def local_1(args):
    state = args["state"]
    owner = state["owner"] if "owner" in state else "local0"

    if state["clientId"] == owner:
        input = args["input"]
        cache = args["cache"]
        cache_dir = state["cacheDirectory"]

        X_file = os.path.join(cache_dir, cache.get("X_filename", ""))
        y_file = os.path.join(cache_dir, cache.get("y_filename", ""))
        with open(X_file, "rb") as fp:
            X = np.load(fp)
        with open(y_file, "rb") as fp:
            y = np.load(fp)
        w_locals = np.array(input["w_locals"])
        intercept_locals = np.array(input["intercept_locals"])

        # split train/test data
        train_split = cache["train_split"] if "train_split" in cache else 0.8
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_split, shuffle=cache["shuffle"]
        )

        # calculate soft predictions u = w'x + intercept based on local classifiers
        # X_train (n_samples, n_features), w_locals(n_features, n_locals), intercept_locals(n_locals,) will be broadcast
        U_train = np.matmul(X_train, w_locals) + intercept_locals
        U_test = np.matmul(X_test, w_locals) + intercept_locals

        # preprocess U_train
        n_samples_owner = U_train.shape[0]
        if cache["fit_intercept_owner"]:
            # add synthetic feature
            synthetic = cache["intercept_scaling_owner"] * np.ones(
                (n_samples_owner, 1)
            )
            U_train_scaled = np.concatenate((U_train, synthetic), axis=1)
        else:
            U_train_scaled = copy.deepcopy(U_train)
        # required by dp: each training sample vector ||x_i|| <= 1
        scale = np.amax(np.linalg.norm(U_train_scaled, axis=1))
        U_train_scaled = U_train_scaled / scale

        # train aggregator model
        w_owner = (1 / scale) * train_model(
            U_train_scaled, y_train, cache, "owner"
        )
        if cache["fit_intercept_owner"]:
            intercept_owner = cache["intercept_scaling_owner"] * w_owner[-1]
            w_owner = w_owner[:-1]
        else:
            intercept_owner = 0.0

        # predict on train/test data, get metrics:
        #   confusion matrix, accuracy, f1-score, ROC AUC
        # confusion matrix: 2D array (n_classes, n_classes)
        # e.g.        predicted
        #               -1    1
        #    true  -1   ...   ...
        #           1   ...   ...
        y_train_pred = predict_linearmodel(w_owner, intercept_owner, U_train)
        cm_train_owner = confusion_matrix(y_train, y_train_pred, normalize=None)
        cm_train_owner_normalized = confusion_matrix(
            y_train, y_train_pred, normalize="true"
        )
        acc_train_owner = accuracy_score(y_train, y_train_pred, normalize=True)

        y_test_pred = predict_linearmodel(w_owner, intercept_owner, U_test)
        cm_test_owner = confusion_matrix(y_test, y_test_pred, normalize=None)
        cm_test_owner_normalized = confusion_matrix(
            y_test, y_test_pred, normalize="true"
        )
        acc_test_owner = accuracy_score(y_test, y_test_pred, normalize=True)
        f1_test_owner = f1_score(y_test, y_test_pred)
        if cache["model_owner"] == "LR":
            y_test_proba = predict_proba_lr(w_owner, intercept_owner, U_test)
        else:  # svm with huber loss
            y_test_proba = predict_decision_svmhuber(
                w_owner, intercept_owner, U_test
            )
        rocauc_test_owner = roc_auc_score(y_test, y_test_proba)

        cm_test_locals = []
        cm_test_locals_normalized = []
        acc_test_locals = []
        f1_test_locals = []
        rocauc_test_locals = []
        for ii in range(intercept_locals.shape[0]):
            y_pred = predict_linearmodel(
                w_locals[:, ii], intercept_locals[ii], X_test
            )
            cm = confusion_matrix(y_test, y_pred, normalize=None).tolist()
            cm_normalized = confusion_matrix(
                y_test, y_pred, normalize="true"
            ).tolist()
            acc = accuracy_score(y_test, y_pred, normalize=True)
            f1 = f1_score(y_test, y_pred)
            if cache["model_local"] == "LR":
                y_proba = predict_proba_lr(
                    w_locals[:, ii], intercept_locals[ii], X_test
                )
            else:  # svm with huber loss
                y_proba = predict_decision_svmhuber(
                    w_locals[:, ii], intercept_locals[ii], X_test
                )
            rocauc = roc_auc_score(y_test, y_proba)
            cm_test_locals.append(cm)
            cm_test_locals_normalized.append(cm_normalized)
            acc_test_locals.append(acc)
            f1_test_locals.append(f1)
            rocauc_test_locals.append(rocauc)

        output_dict = {
            "w_owner": w_owner.tolist(),
            "intercept_owner": float(intercept_owner),
            "cm_train_owner": cm_train_owner.tolist(),
            "cm_train_owner_normalized": cm_train_owner_normalized.tolist(),
            "acc_train_owner": float(acc_train_owner),
            "cm_test_owner": cm_test_owner.tolist(),
            "cm_test_owner_normalized": cm_test_owner_normalized.tolist(),
            "acc_test_owner": float(acc_test_owner),
            "f1_test_owner": float(f1_test_owner),
            "rocauc_test_owner": float(rocauc_test_owner),
            "cm_test_locals": cm_test_locals,
            "cm_test_locals_normalized": cm_test_locals_normalized,
            "acc_test_locals": acc_test_locals,
            "f1_test_locals": f1_test_locals,
            "rocauc_test_locals": rocauc_test_locals,
            "n_samples_owner_train": n_samples_owner,
            "n_samples_owner_test": y_test.shape[0],
            "phase": "local_1",
        }
    else:
        output_dict = {"phase": "local_1"}

    result_dict = {"output": output_dict}
    return json.dumps(result_dict)


if __name__ == "__main__":
    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, "phase"))
    if not phase_key:
        result_dict = local_0(parsed_args)
        sys.stdout.write(result_dict)
    elif "remote_0" in phase_key:
        result_dict = local_1(parsed_args)
        sys.stdout.write(result_dict)
    else:
        raise Exception("Error occurred at Local")
