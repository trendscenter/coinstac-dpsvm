import numpy as np
# from sklearn.linear_model import LogisticRegression  
# from sklearn.linear_model import SGDClassifier  

from .dp_svm import dp_svm
from .dp_lr import dp_lr


def train_model(X, y, input, site='local'):
    '''
    Train a model depicted in input upon data X and y, 
    return coefficients of the model. 

    Keyword arguments:
        X (ndarray of shape (n_sample, n_feature)) --  features
        y (ndarray of shape (n_sample,)) -- label to predict
        input (dict) -- input of COINSTAC pipeline at each iteration
        site (str) -- 'local', 'owner'
    Return:
        w (ndarray of shape (n_feature,)) -- weights in w'x
    '''

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

    ## To run the scikit-learn LR/SVM, comment out the above and 
    ## decomment the below.
    #
    ## scikit-learn LR
    # lambda_ = input['lambda_' + site]
    # clf = LogisticRegression(solver='lbfgs', C=1/lambda_, penalty='l2',
    #                          multi_class='ovr', fit_intercept=False, 
    #                          tol=1e-9, verbose=0, max_iter=100000)
    #
    ## scikit-learn SVM
    # lambda_ = input['lambda_' + site]
    # huberconst=input['huberconst_' + site]
    # clf = SGDClassifier(loss='huber', epsilon=huberconst,
    #                     penalty='l2', alpha=lambda_, 
    #                     fit_intercept=False,
    #                     max_iter=1000, tol=1e-3,
    #                     verbose=0)
    #
    ## predict on training samples                    
    # clf.fit(X, y)
    # w = clf.coef_.flatten()
    # w = w.T
    #
    # pred_y = clf.predict(X)
    # return (w, pred_y)

def predict_linearmodel(weights, X):
    '''
    Return predicted labels for the feature data X.

    Keyword arguments:
        weights (ndarray of shape (n_feature,)) -- weights in w'x
        X (ndarray of shape (n_sample, n_feature)) --  features
    Return:
        labels_pred (ndarray of shape (n_sample,)) -- predicted labels 
    '''
    
    labels_pred = np.where(np.matmul(X, weights) >= 0, 1, -1)
    return labels_pred    