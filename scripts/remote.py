"""Script run at the remote site.

At iteration 0, run remote_0(): Aggregates classifiers from local sites, saves 
    and sends them to the owner site.
At iteration 1, run remote_1(): Receives an aggregator classifier from the 
    owner site and outputs classifiers and perfomance metrics of both owner 
    and local sites.

Raises:
    Exception: If neither 'local_0' nor 'local_1' is in sys.stdin.
"""
import sys

import numpy as np
import ujson as json

from common_functions import list_recursive


def remote_0(args):
    input = args["input"]
    # aggregate from local sites
    w_locals = np.array(
        [
            site_dict["w_local"]
            for site, site_dict in input.items()
            if "w_local" in site_dict
        ]
    ).T

    intercept_locals = np.array(
        [
            site_dict["intercept_local"]
            for site, site_dict in input.items()
            if "intercept_local" in site_dict
        ]
    )

    cm_train_locals = np.array(
        [
            site_dict["cm_train_local"]
            for site, site_dict in input.items()
            if "cm_train_local" in site_dict
        ]
    )

    err_train_locals = np.array(
        [
            site_dict["err_train_local"]
            for site, site_dict in input.items()
            if "err_train_local" in site_dict
        ]
    )

    n_samples_locals = np.array(
        [
            site_dict["n_samples_local"]
            for site, site_dict in input.items()
            if "n_samples_local" in site_dict
        ]
    )

    # dicts
    output_dict = {
        "w_locals": w_locals.tolist(),
        "intercept_locals": intercept_locals.tolist(),
        "phase": "remote_0",
    }

    cache_dict = output_dict.copy()
    cache_dict["cm_train_locals"] = cm_train_locals.tolist()
    cache_dict["err_train_locals"] = err_train_locals.tolist()
    cache_dict["n_samples_locals"] = n_samples_locals.tolist()

    result_dict = {"output": output_dict, "cache": cache_dict}
    return json.dumps(result_dict)


def remote_1(args):
    input = args["input"]
    state = args["state"]
    owner = state["owner"] if "owner" in state else "local0"
    dict_owner = input[owner]
    dict_locals = args["cache"]

    # combine owner and locals
    output_dict = {
        "w_owner": [dict_owner.get("w_owner"), "array"],
        "w_locals": [dict_locals.get("w_locals"), "arrays"],
        "intercept_owner": [dict_owner.get("intercept_owner"), "number"],
        "intercept_locals": [dict_locals.get("intercept_locals"), "array"],
        "cm_test_owner": [dict_owner.get("cm_test_owner"), "table"],
        "cm_test_owner_normalized": [
            dict_owner.get("cm_test_owner_normalized"),
            "table",
        ],
        "cm_train_owner": [dict_owner.get("cm_train_owner"), "table"],
        "cm_test_locals": [dict_owner.get("cm_test_locals"), "tables"],
        "cm_train_locals": [dict_locals.get("cm_train_locals"), "tables"],
        "err_test_owner": [dict_owner.get("err_test_owner"), "number"],
        "err_train_owner": [dict_owner.get("err_train_owner"), "number"],
        "err_test_locals": [dict_owner.get("err_test_locals"), "array"],
        "err_train_locals": [dict_locals.get("err_train_locals"), "array"],
        "n_samples_owner_train": [
            dict_owner.get("n_samples_owner_train"),
            "number",
        ],
        "n_samples_owner_test": [
            dict_owner.get("n_samples_owner_test"),
            "number",
        ],
        "n_samples_locals": [dict_locals.get("n_samples_locals"), "array"],
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
