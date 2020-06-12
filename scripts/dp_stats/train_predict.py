import numpy as np

from .dp_svm import dp_svm
from .dp_lr import dp_lr


def train_model(X, y, input, site='local'):
    # train chosen model depicted in input upon data X and y, return coefficients of the trained model 
    if site != 'owner' and site != 'local':
        raise Exception('Error in train_model: wrong site name')

    if input['model_' + site] == 'LR': 
        w = dp_lr(
            X, y, 
            is_private=input['is_private_' + site],
            perturb_method=input['perturb_method_' + site],
            lambda_=input['lambda_' + site], 
            epsilon=input['epsilon_' + site]
        )
    else:
        w = dp_svm(
            X, y, 
            is_private=input['is_private_' + site],
            perturb_method=input['perturb_method_' + site],
            lambda_=input['lambda_' + site], 
            epsilon=input['epsilon_' + site], 
            huberconst=input['huberconst_' + site]
        )  

    return w

def predict_linearmodel(weights, X):
    # weights is a np array of shape (num_sample, 1)
    # X is a np array of shape (num_sample, num_feature)
    labels_pred = np.where(np.matmul(X, weights) >= 0, 1, -1)
    return labels_pred    