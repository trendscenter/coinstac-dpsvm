import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
import sys
import ujson as json
from ancillary import list_recursive
import common_functions


def remote_1(args):
    input_list = args["input"]
    train_file_name = 'train_data.pickle'
    test_file_name = 'test_data.pickle'

    with open(os.path.join(args["state"]["baseDirectory"], train_file_name),
              'rb') as ftrain:
        train_data = pickle.load(ftrain)

    with open(os.path.join(args["state"]["baseDirectory"], test_file_name),
              'rb') as ftest:
        test_data = pickle.load(ftest)

    W_site = [input_list[site]["W_site"] for site in input_list]

    # train (use public logistic regression classifer to aggregate
    # svm weights from each site)
    sols = []
    train_data_mapped = common_functions.data2data(train_data[0], sols)
    clf = LogisticRegression()
    clf.fit(train_data_mapped, train_data[1])

    raise Exception('hi there')

    # test
    test_data_mapped = common_functions.data2data(test_data[0], sols)
    e = 100 * abs(
        sum(
            map(lambda x: min(0, x),
                clf.predict(test_data_mapped) * test_data[1]))) / float(
                    len(test_data[1]))

#    # also generate the error rate for test_data based on the W_site
#    e_site = []
#    for i in range(0, 4):
#        e_site.append(
#            common_functions.test_errors(
#                np.asarray(sols[i]),
#                np.asarray(test_data[0]), np.asarray(test_data[1])))
#
#    sys.stderr.write(
#        "final error rate of test data using aggregated classifier is: " +
#        str(e) + "\n")
#    sys.stderr.write(
#        "error rates of test data using SVM weights from each site are: ")
#    sys.stderr.write(
#        str(e_site[0]) + ',' + str(e_site[1]) + ',' + str(e_site[2]) + ',' +
#        str(e_site[3]))

    output_dict = {{"final_error_rate": e}}

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
