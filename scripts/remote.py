import sys

import ujson as json

from common_functions import list_recursive


def remote_1(args):
    input_list = args["input"]

    W_site = [
        input_list[site].get('W_site') for site in input_list
        if input_list[site].get('W_site', None)
    ]

    output_dict = {"W_set": W_site, "computation_phase": "remote_1"}

    computation_output = {"output": output_dict}

    return json.dumps(computation_output)


def remote_2(args):
    input_list = args["input"]
    error = [
        input_list[site].get('final_error_rate', 0) for site in input_list
    ]
    output_dict = {"final_error_rate": sum(error)}

    computation_output = {"output": output_dict, "success": True}

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if "local_1" in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    elif "local_2" in phase_key:
        computation_output = remote_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise Exception(parsed_args)
        raise ValueError("Error occurred at Remote")
