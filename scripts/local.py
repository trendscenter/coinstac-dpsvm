"""Script run at the owner site and the local sites.

At iteration 0, run local_0(): Parses data. At the owner site, saves the parsed 
    files to cache. At each local site, trains a classifier and sends it to the
    remote site. 
At iteration 1, run local_1(): At the owner site, trains a classifier, predicts
    for the testing set and sends the classifier and confusion matrix to the 
    remote site. At local sites, nothing.

Raises:
    Exception: If neither 'phase' nor 'remote_0' is in sys.stdin.
"""
import os
import sys

import numpy as np
import ujson as json
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from dp_stats.train_predict import train_model, predict_linearmodel
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
            "model_owner": input["model_owner"],
            "is_private_owner": input["is_private_owner"],
            "perturb_method_owner": input["perturb_method_owner"],
            "lambda_owner": input["lambda_owner"],
            "epsilon_owner": input["epsilon_owner"],
            "huberconst_owner": input["huberconst_owner"],
            "train_split": input["train_split"],
            "X_filename": "X.npy",
            "y_filename": "y.npy",
        }
        output_dict = {"phase": "local_0"}
    else:
        w_local = train_model(X, y, input, "local")
        y_pred = predict_linearmodel(w_local, X)
        cm_local = confusion_matrix(y, y_pred, normalize=None)
        cache_dict = {}
        output_dict = {
            "cm_local": cm_local.tolist(),
            "w_local": w_local.tolist(),
            "num_sample_local": y.shape[0],
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

        # get X, y, W_locals, then calculate soft prediction u = w'x
        X_file = os.path.join(cache_dir, cache.get("X_filename", ""))
        y_file = os.path.join(cache_dir, cache.get("y_filename", ""))
        with open(X_file, "rb") as fp:
            X = np.load(fp)
        with open(y_file, "rb") as fp:
            y = np.load(fp)
        W_locals = np.array(input["W_locals"])
        U = np.matmul(X, W_locals)

        # split train - test data
        train_split = cache["train_split"] if "train_split" in cache else 0.8
        U_train, U_test, y_train, y_test = train_test_split(
            U, y, train_size=train_split, random_state=42, shuffle=True
        )

        # train model, then predict on train and test data,
        # show the results by confusion matrix (cm)
        # cm: 2D array (n_classes, n_classes)
        # e.g.        predicted
        #               -1    1
        #    true  -1   ...   ...
        #           1   ...   ...

        w_owner = train_model(U_train, y_train, cache, "owner")
        y_pred_train = predict_linearmodel(w_owner, U_train)
        cm_train_owner = confusion_matrix(y_train, y_pred_train, normalize=None)

        y_pred = predict_linearmodel(w_owner, U_test)
        cm = confusion_matrix(y_test, y_pred, normalize=None)
        cm_normalized = confusion_matrix(y_test, y_pred, normalize="true")

        output_dict = {
            "cm_train_owner": cm_train_owner.tolist(),
            "cm_owner": cm.tolist(),
            "cm_owner_normalized": cm_normalized.tolist(),
            "w_owner": w_owner.tolist(),
            "num_sample_owner": y.shape[0],
            "phase": "local_1",
        }
    else:
        output_dict = {}

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
