import numpy as np
from scipy.optimize import minimize

from .general_funcs import noisevector


def lrloss(z): 
    hz = np.log(1 + np.exp(-z))
    return hz 

def eval_lr(weights, XY, num, lambda_, b):
    # Return the lr loss.
    
    # add logistic loss from all samples
    XY = np.matmul(XY, weights)
    fw = np.mean(lrloss(XY))

    # add regularization term (1/2 * lambda * |w|^2) and b term (1/n * b'w) 
    fw += 0.5 * lambda_ * weights.dot(weights) + 1.0 / num * b.dot(weights) 
    return fw

def train_lr_nonpriv(XY, num, dim, lambda_):
    # Return the weights of a non-private lr classifier.

    w0 = np.zeros(dim)  # w starting point            
    b = np.zeros(dim)  # zero noise vector
    res = minimize(eval_lr, w0, args=(XY, num, lambda_, b), 
                 method='L-BFGS-B', bounds=None)
    # print('non-priv:')
    # print('    w: ', res.x)
    # print('    status: ', res.success)

    if not res.success:
        raise Exception(res.message)
    w_nonpriv = res.x
    return w_nonpriv

def train_lr_outputperturb(XY, num, dim, lambda_, epsilon):
    # Train a non-private lr classifier, then add noise,
    # return the weights of a private lr classifier.
    # chaudhuri2011differentially: Algorithm 1 output perturbation
    
    w_nonpriv = train_lr_nonpriv(XY=XY, num=num, dim=dim, 
                                 lambda_=lambda_)

    beta = num * lambda_ * epsilon / 2
    noise = noisevector(dim, beta)
    w_priv = w_nonpriv + noise
    # print('output:')
    # print('    w: ', w_priv)

    return w_priv

def train_lr_objectiveperturb(XY, num, dim, lambda_, epsilon):
    # chaudhuri2011differentially: Algorithm 2 objective perturbation
    c = 0.25  # value for lr
    tmp = c / (num * lambda_)
    epsilon_p = epsilon - np.log(1.0 + 2 * tmp + tmp * tmp)
    if epsilon_p < 1e-4:
        raise Exception('Error: Cannot run algorithm' 
                        + 'for this lambda, epsilon and huberconst value')
    
    w0 = np.zeros(dim)
    beta = epsilon_p / 2
    b = noisevector(dim, beta)
    res = minimize(eval_lr, w0, args=(XY, num, lambda_, b),
                   method='L-BFGS-B', bounds=None)
    # print('objective:')
    # print('    w: ', res.x)
    # print('    status: ', res.success)

    if not res.success:
        raise Exception(res.message)
    w_priv = res.x
    return w_priv   

def dp_lr(features, labels, 
          is_private=True, perturb_method='objective', 
          lambda_=0.01, epsilon=0.1):
    '''
    Return a non-private or differentially private lr classifier. 

    Keyword arguments:
        features (ndarray of shape (n_sample, n_feature)) -- X
        labels (ndarray of shape (n_sample,)) -- y
        is_private (bool) -- run private version or not
        perturb_method (str) -- 'output' (output perturbation)
                                 other (objective perturbation)
        lambda_ (float) -- regularization parameter
        epsilon (float) -- privacy parameter

    Return:
        w_nonpriv / w_priv (ndarray of shape (n_feature,)) 
            -- weights in w'x in lr classifier

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
    # lr function only needs [vector x_i * label y_i]
    XY = features * labels[:, np.newaxis]  

    # non-private version
    if not is_private:
        if lambda_ < 0.0:
            raise Exception('ERROR: Lambda should be positive.')
        w_nonpriv = train_lr_nonpriv(XY=XY, num=num, dim=dim, lambda_=lambda_)
        return w_nonpriv

    # private version
    if lambda_ < 0.0 or epsilon < 0.0:
        raise Exception('ERROR: Lambda and Epsilon should all be positive.')

    if perturb_method == 'output':
        w_priv = train_lr_outputperturb(XY=XY, num=num, dim=dim, 
                                        lambda_=lambda_, 
                                        epsilon=epsilon)    
    else:
        w_priv = train_lr_objectiveperturb(XY=XY, num=num, dim=dim,
                                           lambda_=lambda_, 
                                           epsilon=epsilon)
    return w_priv
