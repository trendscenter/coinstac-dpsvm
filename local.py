import dp_stats as dps
import os
import pickle
import sys
import ujson as json
from common_functions import list_recursive


def local_1(args):
    input_list = args["input"]
    file_name = input_list["covariates"]

    with open(os.path.join(args["state"]["baseDirectory"], file_name),
              'rb') as fp:
        data = pickle.load(fp)

    eps = 15
    W_site = dps.dp_svm(data[0], data[1], epsilon=eps)

    output_dict = {"W_site": W_site.tolist(), "computation_phase": "local_1"}
    cache_dict = {}

    computation_output = {'output': output_dict, 'cache': cache_dict}

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
