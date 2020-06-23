"""Script run at the remote site.

At iteration 0, run remote_0(): Aggregates classifiers from local sites, saves 
    and sends them to the owner site.
At iteration 1, run remote_1(): Receives a classifier from the owner site and
    outputs classifiers and confusion matrix from both owner and local sites.

Raises:
    Exception: If neither 'local_0' nor 'local_1' is in sys.stdin.
"""
import sys

import numpy as np
import ujson as json

from common_functions import list_recursive


def remote_0(args):
    input = args["input"]
    # aggregate w_local and num_sample_local from local sites
    W_locals = np.array(
        [
            site_dict["w_local"]
            for site, site_dict in input.items()
            if "w_local" in site_dict
        ]
    ).T

    num_sample_locals = np.array(
        [
            site_dict["num_sample_local"]
            for site, site_dict in input.items()
            if "num_sample_local" in site_dict
        ]
    )

    CM_train_locals = np.array(
        [
            site_dict["cm_local"]
            for site, site_dict in input.items()
            if "cm_local" in site_dict
        ]
    )

    output_dict = {"W_locals": W_locals.tolist(), "phase": "remote_0"}

    cache_dict = output_dict.copy()
    cache_dict["num_sample_locals"] = num_sample_locals.tolist()
    cache_dict["CM_train_locals"] = CM_train_locals.tolist()

    # save a copy in cache
    result_dict = {"output": output_dict, "cache": cache_dict}
    return json.dumps(result_dict)


def remote_1(args):
    input = args["input"]
    locals_dict = args["cache"]
    # extract w_owner and num_sample_owner from the owner site
    for site, site_dict in input.items():
        if site_dict:
            owner_dict = site_dict
            break

    # combine w_owner and W_locals
    output_dict = {
        "w_owner": owner_dict.get("w_owner"),
        "W_locals": locals_dict.get("W_locals"),
        "num_sample_owner": owner_dict.get("num_sample_owner"),
        "num_sample_locals": locals_dict.get("num_sample_locals"),
        "cm_owner": owner_dict.get("cm_owner"),
        "cm_owner_normalized": owner_dict.get("cm_owner_normalized"),
        "cm_train_owner": owner_dict.get("cm_train_owner"),
        "CM_train_locals": locals_dict.get("CM_train_locals"),
    }

    result_dict = {"output": output_dict, "success": True}
    return json.dumps(result_dict)


if __name__ == "__main__":
    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, "phase"))

    if "local_0" in phase_key:
        result_dict = remote_0(parsed_args)
        sys.stdout.write(result_dict)
    elif "local_1" in phase_key:
        result_dict = remote_1(parsed_args)
        sys.stdout.write(result_dict)
    else:
        raise Exception("Error occurred at Remote")
