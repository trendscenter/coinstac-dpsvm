import numpy as np
import os
import sys

import ujson as json

import common_functions
import dp_stats as dps
from common_functions import list_recursive
from sklearn.linear_model import LogisticRegression


def local_1(args):
    input_list = args["input"]

    try:
        train_data = input_list["covariates"][0]
        train_labels = input_list["covariates"][1]
        test_data = input_list["covariates"][2]
        test_labels = input_list["covariates"][3]
        cache_dict = {
            "train_data": train_data,
            "train_labels": train_labels,
            "test_data": test_data,
            "test_labels": test_labels
        }
    except IndexError:
        cache_dict = {}

    input_dir = os.path.join(args["state"]["baseDirectory"])
    num_files = len([
        name for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
    ])

    if num_files == 2:
        with open(os.path.join(args["state"]["baseDirectory"], train_data),
                  'rb') as fp:
            train_data = np.load(fp)
        with open(os.path.join(args["state"]["baseDirectory"], train_labels),
                  'rb') as fp:
            train_labels = np.load(fp)

        eps = 10
        W_site = dps.dp_svm(train_data, train_labels, epsilon=eps)

        output_dict = {
            "W_site": W_site.tolist(),
            "computation_phase": "local_1"
        }
    else:
        output_dict = {"computation_phase": "local_1"}

    computation_output = {'output': output_dict, 'cache': cache_dict}

    return json.dumps(computation_output)


def local_2(args):
    W_site = args["input"]["W_set"]
    train_data = args["cache"].get('train_data', "")
    train_labels = args["cache"].get('train_labels', "")
    test_data = args["cache"].get('test_data', "")
    test_labels = args["cache"].get('test_labels', "")

    full_train_data = os.path.join(args["state"]["baseDirectory"], train_data)
    full_train_labels = os.path.join(args["state"]["baseDirectory"],
                                     train_labels)

    full_test_data = os.path.join(args["state"]["baseDirectory"], test_data)
    full_test_labels = os.path.join(args["state"]["baseDirectory"],
                                    test_labels)

    if os.path.isfile(full_train_data) and os.path.isfile(
            full_test_data) and os.path.isfile(
                full_train_labels) and os.path.isfile(full_test_labels):

        with open(full_train_data, 'rb') as fp:
            X_train = np.load(fp)

        with open(full_train_labels, 'rb') as fp:
            y_train = np.load(fp)

        with open(full_test_data, 'rb') as fp:
            X_test = np.load(fp)

        with open(full_test_labels, 'rb') as fp:
            y_test = np.load(fp)

        # train (use public logistic regression classifer to aggregate
        # svm weights from each site)
        train_data_mapped = common_functions.data2data(X_train, W_site)
        clf = LogisticRegression(solver='lbfgs')
        clf.fit(train_data_mapped, y_train)
        
        # test
        test_data_mapped = common_functions.data2data(X_test, W_site)
        e = 100 * abs(
            sum(
                map(lambda x: min(0, x),
                    clf.predict(test_data_mapped) * y_test))) / len(y_test)

        output_dict = {"final_error_rate": e, "computation_phase": "local_2"}

    else:
        output_dict = {}

    computation_output = {"output": output_dict}

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))
    if not phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_1" in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
