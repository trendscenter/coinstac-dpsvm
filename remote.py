import ujson as json
from sklearn.linear_model import LogisticRegression
import numpy as np
import sys
from ancillary import list_recursive
from common_functions import *


def remote_1(args):
    input_list = args["input"]
    file_name = 'data.pickle'

    with open(os.path.join(args["state"]["baseDirectory"], file_name),
              'rb') as fp:
        train_data = pickle.load(fp)

    # train (use public logistic regression classifer to aggregate svm weights from each site)
    train_data_mapped = data2data(train_data[0], sols)
    clf = LogisticRegression()
    clf.fit(train_data_mapped, train_data[1])

    #test
    test_data_mapped = data2data(test_data[0], sols)
    e = 100 * abs(
        sum(
            map(lambda x: min(0, x),
                clf.predict(test_data_mapped) * test_data[1]))) / double(
                    len(test_data[1]))

    #also generate the error rate for test_data based on the W_site
    e_site = []

    for i in range(0, 4):
        e_site.append(
            test_errors(
                np.asarray(sols[i]),
                np.asarray(test_data[0]), np.asarray(test_data[1])))

    sys.stderr.write(
        "final error rate of test data using aggregated classifier is: " +
        str(e) + "\n")
    sys.stderr.write(
        "error rates of test data using SVM weights from each site are: ")
    sys.stderr.write(
        str(e_site[0]) + ',' + str(e_site[1]) + ',' + str(e_site[2]) + ',' +
        str(e_site[3]))

    output_dict = {{"final_error_rate": e}}

    computation_output = {"output": output_dict, "success": True}

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if "local_1" in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
#    elif "local_2" in phase_key:
#        computation_output = remote_2(parsed_args)
#        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
