import numpy as np
#from statistics import stdev
# from pylab import norm
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
    # add hinge loss: max(0, 1 - y_iw'x) from all samples
    # XY = np.matmul(XY, weights)
    # fw = np.mean(np.where(XY > 1.0, 0, 1 - XY))   

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
    # return a non-private svm classifier

    w0 = np.zeros(dim)  # w starting point              ?? dataframe
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

def train_svm_outputperturbation(XY, num, dim, lambda_, epsilon, huberconst):
    # train a non-private svm classifier, then add noise
    # chaudhuri2011differentially: Algorithm 1 ERM with output perturbation
    
    w_nonpriv = train_svm_nonpriv(XY=XY, num=num, dim=dim, lambda_=lambda_, huberconst=huberconst) 

    beta = num * lambda_ * epsilon / 2
    noise = noisevector(dim, beta)
    w_priv = w_nonpriv + noise

    # print('output:')
    # print('    w: ', w_priv)
    return w_priv

def train_svm_objectiveperturbation(XY, num, dim, lambda_, epsilon, huberconst):
    # chaudhuri2011differentially: Algorithm 2 ERM with objective perturbation

    c = 1 / (2 * huberconst)  # value for svm 
    tmp = c / (num * lambda_)
    epsilon_p = epsilon - np.log(1.0 + 2 * tmp + tmp * tmp)

    # raise Exception('XY:' + str(XY.shape) +'\n'
    #                 + 'num:' + str(num) + '\n'
    #                 + 'dim:' + str(dim) + '\n'
    #                 + 'lambda:' + str(lambda_) + '\n'
    #                 + 'epsilon: ' + str(epsilon) + '\n'
    #                 + 'h: ' + str(huberconst) + '\n'
    #                 + 'c: ' + str(c) + '\n'
    #                 + 'tmp: ' + str(tmp) + '\n'
    #                 + 'epsilon_p: ' + str(epsilon_p) + '\n'
    #     )

    if epsilon_p < 1e-4:
        raise Exception('Error: Cannot run algorithm for this lambda, epsilon and huberconst value')
    
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

def dp_svm(features, labels, is_private=True, perturb_method='objective', lambda_=0.01, epsilon=0.1, huberconst=0.5):
    '''This function provides a differentially-private estimate of the svm classifier according to Sarwate et al. 2011,
    'Differentially Private Empirical Risk Minimization' paper.

    Input:

      features = features matrix, samples are in rows
      labels = labels of the data samples
      method = 'obj' (for objective perturbation) or 'out' (for output perturbation)
      epsilon = privacy parameter, default 1.0
      lambda_ = regularization parameter
      h = huber loss parameter

    Output:

      fpriv = epsilon-differentially-private estimate of the svm classifier

    Example:

      >>> import numpy as np
      >>> import dp_stats as dps

      >>> X = np.random.normal(1.0, 1.0, (n,d));
      >>> Y = np.random.normal(-1.0, 1.0, (n,d));
      >>> labelX = 1.0 * np.ones(n);
      >>> labelY = -1.0 * np.ones(n);

      >>> features = np.vstack((X,Y));
      >>> labels = np.hstack((labelX,labelY));

      >>> fpriv = dps.classification.dp_svm (features, labels, 'obj', 0.1, 0.01, 0.5)
      [  9.23418189   2.63380995  -2.01654661  -1.19112074  17.32083386
         3.37943017 -14.76815378  12.3119061   -1.82132988  24.03559848]
    '''

    num = features.shape[0]  # number of samples
    dim = features.shape[1]  # dimension of a sample vector 
    XY = features * labels[:, np.newaxis] # svm function only needs [vector x_i * label y_i]

    # non-private version
    if not is_private:
        if lambda_ < 0.0 or huberconst < 0.0:
            raise Exception('ERROR: Lambda and Huberconst should all be positive.')
        w_nonpriv = train_svm_nonpriv(XY=XY, num=num, dim=dim, lambda_=lambda_, huberconst=huberconst) 
        return w_nonpriv

    # private version
    if lambda_ < 0.0 or epsilon < 0.0 or huberconst < 0.0:
        raise Exception('ERROR: Lambda, Epsilon and Huberconst should all be positive.')
    
    if perturb_method == 'objective':
        w_priv = train_svm_objectiveperturbation(XY=XY, num=num, dim=dim, lambda_=lambda_, epsilon=epsilon, huberconst=huberconst)
    else:
        w_priv = train_svm_outputperturbation(XY=XY, num=num, dim=dim, lambda_=lambda_, epsilon=epsilon, huberconst=huberconst)
    return w_priv

    # svm function only needs [vector x_i * label y_i], i.e., multiply all the training data vectors by their labels
    # XY = features.multiply(labels[labels.columns[0]], axis='index')
    # XY = XY.to_numpy(copy=False) 

    # train_svm_objectiveperturbation(XY, num, dim, lambda_, epsilon, huberconst)