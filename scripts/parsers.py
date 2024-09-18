"""Parsers to extract features X and labels y from inputspec.json.
"""
import os

import numpy as np
import pandas as pd


def parse_covar_info(covar_info):
    # convert bool to categorical as soon as possible
    for column in covar_info.select_dtypes(bool):
        covar_info[column] = covar_info[column].astype("object")

    covar_info = covar_info.apply(pd.to_numeric, errors="ignore")

    # convert contents of object columns to lowercase
    for column in covar_info.select_dtypes(object):
        covar_info[column] = covar_info[column].astype("str").str.lower()

    return covar_info


def parse_for_categorical(X_df):
    """Return unique subsites as a dictionary"""
    X = parse_covar_info(X_df)

    site_dict1 = {col: list(X[col].unique()) for col in X.select_dtypes(include=object)}

    return X, site_dict1

def fsl_parser(input, base_dir):
    """Parses fsl-specific inputspec.json, returns features X and labels y.

    If there is no fsl section in inputspec.json, then only the csv section
    will be parsed.

    Args:
        input (dict): Input of COINSTAC pipeline at each iteration.
        base_dir (str): baseDirectory at each site.

    Returns:
        features(pandas df of shape (n_sample, n_feature)): X.
        labels_np (ndarray of shape (n_sample,)): y.
    """
    # process covariates (csv section)
    '''
    covariates_raw = input["covariates"]
    covariates = covariates_raw[0][0][
        1:
    ]  # e.g., [[subject0.txt, true, 47], ...]
    covariate_names = covariates_raw[0][0][
        0
    ]  # e.g., freesurferfile, isControl, age
    index_name = covariates_raw[0][0][0][0]  # e.g., freesurferfile
    '''
    index_name = 'freesurferfile'
    X_info = input["covariates"]
    y_info = input["data"]
    X_df = pd.DataFrame(X_info).T
    X_df = pd.DataFrame(X_df, columns=X_df.columns, index=X_df.index)
    X_df.reset_index(inplace=True)
    X_df = X_df.rename(columns={'index': index_name})


    y_labels = y_info[0]["value"]


    # process measurements (fsl section)
    measurement_files = X_df[index_name].tolist()  # e.g., ['subject0.txt', ...]

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

    Y_df = pd.concat(tmp_list)
    Y_df = pd.DataFrame(Y_df)
    Y_df.set_index(index_name, inplace=True)

    #Filter freesurfer ROIs from stats file based on the provided input ROIs
    if len(y_labels)>0:
        missing_fs_stats = [_temp for _temp in y_labels if _temp not in Y_df.columns.to_list()]
        assert len(missing_fs_stats) == 0, f'Missing freesurfer stats: {str(missing_fs_stats)}'
        Y_df = Y_df[np.intersect1d(Y_df.columns, y_labels)]

    # merge covariates and measurements into features_df
    X_df = pd.merge(
        Y_df, X_df, on=index_name, how='inner', validate='one_to_one'
    )

    X_df.set_index(index_name, inplace=True)

    # separate X_df to get features matrix and labels matrix
    # Note:
    #     if a categorical class is read in as int or as numerical
    #     string, e.g.,'1', then you should manually set this class
    #     as categorical type before calling pd.get_dummies():
    #         df['A'] = df['A'].astype('category')
    #     Otherwise, pd.to_numeric converts is to int/float.
    label_name = input["label"]
    labels_df = X_df.pop(label_name)

    labels_df = labels_df.apply(pd.to_numeric, errors="ignore")
    labels_df = pd.get_dummies(labels_df, drop_first=True, dtype=np.int8)
    labels_df = labels_df * 1
    labels_df = labels_df.apply(pd.to_numeric, errors="ignore")
    labels_df.replace(
        0, -1, inplace=True
    )  # binary label [0, 1] -> [-1, 1] for LR and SVM

    labels_np = labels_df.to_numpy().flatten()

    return (X_df, labels_df)

