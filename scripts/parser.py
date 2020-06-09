#!/usr/bin/env python3
import os
import sys
import pandas as pd
import json

def fsl_parser(input):
    '''parse fsl-specific inputspec.json, return features matrix and labels matrix'''
    # process covariates
    covariates_raw = input["covariates"]
    covariates = covariates_raw["value"][0][0][1:]
    covariate_names = covariates_raw["value"][0][0][0]  # e.g., isControl, age
    index_name = covariates_raw["value"][0][0][0][0]  # e.g., "subject0_aseg_stats.txt"
    features_df = pd.DataFrame.from_records(covariates, columns=covariate_names)
    features_df.set_index(index_name, inplace=True)

    # process measurements 
    measurements_raw = input["data"]
    measurement_files = measurements_raw["value"][0]
    # # measurement_names = measurements_raw["value"][2]

    tmp_list = []
    for file in measurement_files:
        print(file)
        _x = pd.read_csv(os.path.join("/home/hju/ssd/repo/coinstac-dpsvm/test/input/local0/simulatorRun",
                                      file),
                        sep='\t',
                        skiprows=[0],
                        header=None,
                        index_col=0).transpose()
        _x.insert(loc=0, column=index_name, value=file)
        tmp_list.append(_x)

    measurement_df = pd.concat(tmp_list)
    measurement_df.set_index(index_name, inplace=True)

    # merge covariates and measurements into features_df
    features_df = features_df.merge(measurement_df,
                                    how='inner',
                                    left_index=True,
                                    right_index=True)

    features_df = features_df.apply(pd.to_numeric, errors='ignore')
    features_df = pd.get_dummies(features_df, drop_first=True)
    features_df = features_df * 1  # True -> 1, False -> 0
    features_df = features_df.apply(pd.to_numeric, errors='ignore')  # object 1, 0 -> int
    features_df.replace(0, -1, inplace=True)  
    
    # get features matrix and labels matrix
    label_name = input["label"]["value"]
    labels_df = features_df.pop(label_name)
    # labels_df.replace(0, -1, inplace=True)
    return (features_df, labels_df)


with open("scrach.json", "r") as inputspec:
    input = json.load(inputspec)

features, labels = fsl_parser(input)
# print()
# print(labels)
# print(labels.dtypes)
# print()
# print(features)