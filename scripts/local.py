import os
import sys

import numpy as np
import ujson as json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import common_functions
import dp_stats as dps
from common_functions import list_recursive
from parsers import fsl_parser


def local_1(args):
    state_ = args["state"]
    cache_dir = state_["cacheDirectory"]

    (X, y) = fsl_parser(args)

    y_temp = y
    y = X['isControl']
    y = y.replace(0, -1)
    X = X.drop(columns=['isControl'])

    X = y_temp.merge(X, how='inner', left_index=True, right_index=True)
    X = X.drop(columns=['EstimatedTotalIntraCranialVol'])

    eps = 10

    if args['state']['clientId'] != 'local0':
        W_site = dps.dp_svm(X.values, y.values, epsilon=eps)
        cache_dict = {}
        output_dict = {
            "W_site": W_site.tolist(),
            "computation_phase": "local_1"
        }
    else:
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.33,
                                                            random_state=42)

        np.save(os.path.join(cache_dir, 'X_test.npy'), X_test.values)
        np.save(os.path.join(cache_dir, 'y_test.npy'), y_test.values)
        np.save(os.path.join(cache_dir, 'X_train.npy'), X_train.values)
        np.save(os.path.join(cache_dir, 'y_train.npy'), y_train.values)

        cache_dict = {
            "X_train": 'X_train.npy',
            "y_train": 'y_train.npy',
            "X_test": 'X_test.npy',
            "y_test": 'y_test.npy'
        }
        output_dict = {"computation_phase": "local_1"}

    computation_output = {'output': output_dict, 'cache': cache_dict}

    return json.dumps(computation_output)


def local_2(args):
    W_site = args["input"]["W_set"]
    train_data = args["cache"].get('X_train', "")
    train_labels = args["cache"].get('y_train', "")
    test_data = args["cache"].get('X_test', "")
    test_labels = args["cache"].get('y_test', "")

    full_train_data = os.path.join(args["state"]["cacheDirectory"], train_data)
    full_train_labels = os.path.join(args["state"]["cacheDirectory"],
                                     train_labels)

    full_test_data = os.path.join(args["state"]["cacheDirectory"], test_data)
    full_test_labels = os.path.join(args["state"]["cacheDirectory"],
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
