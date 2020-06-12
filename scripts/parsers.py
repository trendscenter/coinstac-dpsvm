import os
# import sys
import pandas as pd
import ujson as json
import numpy as np

def fsl_parser(input, base_dir):
    '''parse fsl-specific inputspec.json, return features matrix and labels matrix as np arrays'''

    # process covariates
    covariates_raw = input['covariates']

    # with open('/home/hju/ssd/repo/coinstac-dpsvm/scripts/debug.txt', 'w+') as f:
    #     f.write(str(type(covariates_raw)))
    #     f.write(str(len(covariates_raw)))
    #     f.write(str(covariates_raw[0]))
    # raise Exception(str(covariates_raw[0]))

    covariates = covariates_raw[0][0][1:]  # e.g., [[subject0.txt, true, 47], ...]
    covariate_names = covariates_raw[0][0][0]  # e.g., freesurferfile, isControl, age
    index_name = covariates_raw[0][0][0][0]  # e.g., freesurferfile
    # raise Exception(str(covariate_names) + '\n' + str(covariates[-1]))

    features_df = pd.DataFrame.from_records(covariates, columns=covariate_names)
    features_df.set_index(index_name, inplace=True)

    # process measurements 
    measurements_raw = input['measurements']
    measurement_files = measurements_raw[0]  # e.g., ['subject0.txt', ...]

    tmp_list = []
    for file in measurement_files:
        print(file)
        _x = pd.read_csv(os.path.join(base_dir, file),
            sep='\t',
            skiprows=[0],
            header=None,
            index_col=0
        ).transpose()
        _x.insert(loc=0, column=index_name, value=file)
        tmp_list.append(_x)

    measurement_df = pd.concat(tmp_list)
    measurement_df.set_index(index_name, inplace=True)

    # merge covariates and measurements into features_df
    features_df = features_df.merge(measurement_df,
        how='inner',
        left_index=True,
        right_index=True
    )

    features_df = features_df.apply(pd.to_numeric, errors='ignore')
    features_df = pd.get_dummies(features_df, drop_first=True)
    features_df = features_df * 1  # True -> 1, False -> 0
    features_df = features_df.apply(pd.to_numeric, errors='ignore')  # object 1, 0 -> int ??
    # features_df.replace(0, -1, inplace=True)  
    
    # separate features_df to get features matrix and labels matrix
    label_name = input['label']
    labels_df = features_df.pop(label_name)
    labels_df.replace(0, -1, inplace=True)  # binary label [0, 1] -> [-1, 1] for LR and SVM

    features_np = features_df.to_numpy()
    labels_np = labels_df.to_numpy()
    # raise Exception(str(features_np) + '\n\n' + str(labels_np) + 
    #                     '\n\n' + str(features_np.shape) +'\n\n'+str(labels_np.shape))

    # normalize features to ensure ||x_i|| <= 1
    max_norm = np.amax(np.linalg.norm(features_np, axis=1))
    features_np = features_np / max_norm

    return (features_np, labels_np)



# with open(inputspec_file, 'r') as inputspec:
#     input = json.load(inputspec)

# features, labels = fsl_parser(input)

# print()
# print(labels)
# print(labels.dtypes)
# print()
# print(features)