"""Parsers to extract features X and labels y from raw data.
"""
import os

import numpy as np
import pandas as pd


def fsl_parser(input, base_dir):
    """Parses fsl-specific inputspec.json, returns features X and labels y.

    Args:
        input (dict): Input of COINSTAC pipeline at each iteration.
        base_dir (str): baseDirectory at each site.

    Returns:
        features_np (ndarray of shape (n_sample, n_feature)): X.
        labels_np (ndarray of shape (n_sample,)): y.
    """
    # process covariates
    covariates_raw = input["covariates"]
    covariates = covariates_raw[0][0][
        1:
    ]  # e.g., [[subject0.txt, true, 47], ...]
    covariate_names = covariates_raw[0][0][
        0
    ]  # e.g., freesurferfile, isControl, age
    index_name = covariates_raw[0][0][0][0]  # e.g., freesurferfile

    features_df = pd.DataFrame.from_records(covariates, columns=covariate_names)
    features_df.set_index(index_name, inplace=True)

    # process measurements (data)
    measurements_raw = input["data"]
    measurement_files = measurements_raw[0]  # e.g., ['subject0.txt', ...]

    tmp_list = []
    for file in measurement_files:
        _x = pd.read_csv(
            os.path.join(base_dir, file),
            sep="\t",
            skiprows=[0],
            header=None,
            index_col=0,
        ).transpose()
        _x.insert(loc=0, column=index_name, value=file)
        tmp_list.append(_x)

    measurement_df = pd.concat(tmp_list)
    measurement_df.set_index(index_name, inplace=True)

    # merge covariates and measurements into features_df
    features_df = features_df.merge(
        measurement_df, how="inner", left_index=True, right_index=True
    )

    # separate features_df to get features matrix and labels matrix
    # Note:
    #     if a categorical class is read in as int or as numerical
    #     string, e.g.,'1', then you should manually set this class
    #     as categorical type before calling pd.get_dummies():
    #         df['A'] = df['A'].astype('category')
    #     Otherwise, pd.to_numeric converts is to int/float.
    label_name = input["label"]
    labels_df = features_df.pop(label_name)

    # process categorical classes, then convert dataframes to numpy arrays
    features_df = features_df.apply(pd.to_numeric, errors="ignore")
    features_df = pd.get_dummies(features_df, drop_first=True)
    features_df = features_df * 1  # True -> 1, False -> 0
    features_df = features_df.apply(
        pd.to_numeric, errors="ignore"
    )  # object 0, 1 -> int

    labels_df = labels_df.apply(pd.to_numeric, errors="ignore")
    labels_df = pd.get_dummies(labels_df, drop_first=True, dtype=np.int8)
    labels_df = labels_df * 1
    labels_df = labels_df.apply(pd.to_numeric, errors="ignore")
    labels_df.replace(
        0, -1, inplace=True
    )  # binary label [0, 1] -> [-1, 1] for LR and SVM

    features_np = features_df.to_numpy()
    labels_np = labels_df.to_numpy().flatten()

    # normalize features to ensure ||x_i|| <= 1
    max_norm = np.amax(np.linalg.norm(features_np, axis=1))
    features_np = features_np / max_norm

    return (features_np, labels_np)
