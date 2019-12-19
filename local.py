import os
import pickle
import sys

import ujson as json

import common_functions
import dp_stats as dps
from common_functions import list_recursive
from sklearn.linear_model import LogisticRegression


def local_1(args):
    input_list = args["input"]
    file_name = input_list["covariates"][0]
    
    try:
        train_file_name = input_list["covariates"][1]
        test_file_name = input_list["covariates"][2]
        cache_dict = {"train_file": train_file_name, "test_file": test_file_name}
    except IndexError:
        cache_dict = {}

    with open(os.path.join(args["state"]["baseDirectory"], file_name),
              'rb') as fp:
        data = pickle.load(fp)

    eps = 15
    W_site = dps.dp_svm(data[0], data[1], epsilon=eps)

    output_dict = {"W_site": W_site.tolist(), "computation_phase": "local_1"}


    computation_output = {'output': output_dict, 'cache': cache_dict}
    return json.dumps(computation_output)


def local_2(args):
    W_site = args["input"]["W_set"]
    train_file_name = args["cache"].get('train_file', "")
    test_file_name = args["cache"].get('test_file', "")

    full_train_file = os.path.join(args["state"]["baseDirectory"], train_file_name)
    full_test_file = os.path.join(args["state"]["baseDirectory"], test_file_name)
                                   
    if os.path.exists(full_train_file) and os.path.exists(full_test_file):

        with open(full_train_file, 'rb') as ftrain:
            X_train, y_train = pickle.load(ftrain)
    
        with open(full_test_file,
                  'rb') as ftest:
            X_test, y_test = pickle.load(ftest)
    
        # train (use public logistic regression classifer to aggregate
        # svm weights from each site)
        train_data_mapped = common_functions.data2data(X_train, W_site)
        clf = LogisticRegression(solver='lbfgs')
        clf.fit(train_data_mapped, y_train)
    
        # test
        test_data_mapped = common_functions.data2data(X_test, W_site)
        e = 100 * abs(
            sum(map(lambda x: min(0, x),
                    clf.predict(test_data_mapped) * y_test))) / len(y_test)

        output_dict = {"final_error_rate": e}
    
    else:
        output_dict = {}

    computation_output = {"output": output_dict, "computation_phase": "local_2"}

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
