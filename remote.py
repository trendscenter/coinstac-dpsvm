import os
import pickle
from sklearn.linear_model import LogisticRegression
import sys
import ujson as json
import common_functions
from common_functions import list_recursive


def remote_1(args):
    input_list = args["input"]
    train_file_name = 'train_data.pickle'
    test_file_name = 'test_data.pickle'

    with open(os.path.join(args["state"]["baseDirectory"], train_file_name),
              'rb') as ftrain:
        X_train, y_train = pickle.load(ftrain)

    with open(os.path.join(args["state"]["baseDirectory"], test_file_name),
              'rb') as ftest:
        X_test, y_test = pickle.load(ftest)

    # train (use public logistic regression classifer to aggregate
    # svm weights from each site)
    W_site = [input_list[site]['W_site'] for site in input_list]

    train_data_mapped = common_functions.data2data(X_train, W_site)
    clf = LogisticRegression()
    clf.fit(train_data_mapped, y_train)

    # test
    test_data_mapped = common_functions.data2data(X_test, W_site)
    e = 100 * abs(
        sum(
            map(lambda x: min(0, x),
                clf.predict(test_data_mapped) * y_test))) / len(y_test)

    output_dict = {"final_error_rate": e}

    computation_output = {"output": output_dict, "success": True}

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if "local_1" in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
