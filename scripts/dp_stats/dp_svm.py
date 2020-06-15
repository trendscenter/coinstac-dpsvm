import numpy as np
from scipy.optimize import minimize

from .general_funcs import noisevector


def huberloss(z, huberconst):  
    # chaudhuri2011differentially: equation 7 & corollary 13
    if z > 1.0 + huberconst:
        hz = 0
    elif z < 1.0 - huberconst:
        hz = 1 - z
    else:
        hz = (1 + huberconst - z)**2 / (4 * huberconst)
    return hz

def eval_svm(weights, XY, num, lambda_, b, huberconst):
    # Return the svm loss.

    # add huber loss from all samples
    XY = np.matmul(XY, weights)
    fw = 0
    for z in XY:
        fw += huberloss(z=z, huberconst=huberconst)
    fw /= float(num) 

    # add regularization term (1/2 * lambda * |w|^2) and b term (1/n * b'w) 
    fw += 0.5 * lambda_ * weights.dot(weights) + 1.0 / num * b.dot(weights) 
    return fw

def train_svm_nonpriv(XY, num, dim, lambda_, huberconst):
    # Return the weights of a non-private svm classifier.

    w0 = np.zeros(dim)  # w starting point           
    b = np.zeros(dim)  # zero noise vector
    res = minimize(eval_svm, w0, args=(XY, num, lambda_, b, huberconst), 
                   method='L-BFGS-B', bounds=None)
    # print('non-priv:')
    # print('    w: ', res.x)
    # print('    status: ', res.success)

    if not res.success:
        raise Exception(res.message)
    w_nonpriv = res.x
    return w_nonpriv

def train_svm_outputperturb(XY, num, dim, lambda_, epsilon, huberconst):
    # Train a non-private svm classifier, then add noise, 
    # return the weights of a private svm classifier.
    # chaudhuri2011differentially: Algorithm 1 output perturbation

    w_nonpriv = train_svm_nonpriv(XY=XY, num=num, dim=dim, 
                                  lambda_=lambda_, 
                                  huberconst=huberconst) 

    beta = num * lambda_ * epsilon / 2
    noise = noisevector(dim, beta)
    w_priv = w_nonpriv + noise
    # print('output:')
    # print('    w: ', w_priv)

    return w_priv

def train_svm_objectiveperturb(XY, num, dim, lambda_, epsilon, huberconst):
    # Return the weights of a private svm classifier.
    # chaudhuri2011differentially: Algorithm 2 objective perturbation
    # http://cseweb.ucsd.edu/~kamalika/code/dperm/documentation.pdf

    c = 1 / (2 * huberconst)  # value for svm 
    tmp = c / (num * lambda_)
    epsilon_p = epsilon - np.log(1.0 + 2 * tmp + tmp * tmp)

    if epsilon_p < 1e-4:
        raise Exception('Error: Cannot run algorithm' 
                        + 'for this lambda, epsilon and huberconst value')
    
    w0 = np.zeros(dim)
    beta = epsilon_p / 2
    b = noisevector(dim, beta)
    res = minimize(eval_svm, w0, args=(XY, num, lambda_, b, huberconst),
                   method='L-BFGS-B', bounds=None)
    # print('objective:')
    # print('    w: ', res.x)
    # print('    status: ', res.success)

    if not res.success:
        raise Exception(res.message)
    w_priv = res.x
    return w_priv   

def dp_svm(features, labels, 
           is_private=True, perturb_method='objective', 
           lambda_=0.01, epsilon=0.1, huberconst=0.5):
    '''
    Return a non-private or differentially private svm classifier. 

    Keyword arguments:
        features (ndarray of shape (n_sample, n_feature)) -- X
        labels (ndarray of shape (n_sample,)) -- y
        is_private (bool) -- run private version or not
        perturb_method (str) -- 'output' (output perturbation)
                                 other (objective perturbation)
        lambda_ (float) -- regularization parameter
        epsilon (float) -- privacy parameter
        huberconst (float) = huber loss parameter

    Return:
        w_nonpriv / w_priv (ndarray of shape (n_feature,)) 
            -- weights in w'x in svm classifier

    Reference:
        [1] K. Chaudhuri, C. Monteleoni, and A. D. Sarwate, 
        “Differentially  privateempirical risk minimization,” 
        Journal of Machine Learning Research, vol. 12,no. Mar, 
        pp. 1069–1109, 2011.
        [2] ——, “Documentation for regularized lr and regularized svm 
        code,” Available at 
        http://cseweb.ucsd.edu/kamalika/code/dperm/documentation.pdf.
    '''

    num = features.shape[0]  # number of samples
    dim = features.shape[1]  # dimension of a sample vector x_i
    # svm function only needs [vector x_i * label y_i]
    XY = features * labels[:, np.newaxis]  

    # non-private version
    if not is_private:
        if lambda_ < 0.0 or huberconst < 0.0:
            raise Exception('ERROR: Lambda and Huberconst'
                            + 'should all be positive.')
        w_nonpriv = train_svm_nonpriv(XY=XY, num=num, dim=dim, 
                                      lambda_=lambda_, 
                                      huberconst=huberconst) 
        return w_nonpriv

    # private version
    if lambda_ < 0.0 or epsilon < 0.0 or huberconst < 0.0:
        raise Exception('ERROR: Lambda, Epsilon and Huberconst' 
                        + 'should all be positive.')
    
    if perturb_method == 'output':
        w_priv = train_svm_outputperturb(XY=XY, num=num, dim=dim, 
                                         lambda_=lambda_, 
                                         epsilon=epsilon, 
                                         huberconst=huberconst)
    else:
        w_priv = train_svm_objectiveperturb(XY=XY, num=num, dim=dim, 
                                            lambda_=lambda_, 
                                            epsilon=epsilon, 
                                            huberconst=huberconst)
    return w_priv
