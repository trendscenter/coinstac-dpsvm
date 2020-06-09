import numpy as np
#from statistics import stdev
# from pylab import norm
from scipy.optimize import minimize
from dp_func import noisevector


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
        
    # add huber loss from all samples ( 1/n * sum( huberloss(y_i w' x_i) ) )
    # fw = average( huberloss(XY.dot(weights)) )
    XY = np.matmul(XY, weights)
    fw = 0
    for z in XY:
        fw += huberloss(z, huberconst)
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

    print("non-priv:")
    print("    w: ", res.x)
    print("    status: ", res.success)

    # if not res.success:
    #     raise Exception(res.message)
    f = res.x
    return f

def train_svm_outputperturbation(XY, num, dim, lambda_, epsilon, huberconst):
    # train a non-private svm classifier, then add noise
    # chaudhuri2011differentially: Algorithm 1 ERM with output perturbation
    
    f = train_svm_nonpriv(XY, num, dim, lambda_, huberconst)

    beta = num * lambda_ * epsilon / 2
    noise = noisevector(dim, beta)
    fpriv = f + noise

    print("output:")
    print("    w: ", fpriv)
    return fpriv

def train_svm_objectiveperturbation(XY, num, dim, lambda_, epsilon, huberconst):
    # chaudhuri2011differentially: Algorithm 2 ERM with objective perturbation

    c = 1 / (2 * huberconst)  # value for svm 
    tmp = c / (num * lambda_)
    epsilon_p = epsilon - np.log(1.0 + 2 * tmp + tmp * tmp)
    if epsilon_p < 1e-4:
        raise Exception("Error: Cannot run algorithm for this lambda, epsilon and huberconst value")
    
    w0 = np.zeros(dim)
    beta = epsilon_p / 2
    b = noisevector(dim, beta)
    res = minimize(eval_svm, w0, args=(XY, num, lambda_, b, huberconst),
                   method='L-BFGS-B', bounds=None)

    print("objective:")
    print("    w: ", res.x)
    print("    status: ", res.success)

    # if not res.success:
    #     raise Exception(res.message)
    fpriv = res.x
    return fpriv   

def dp_svm(features, labels, lambda_=0.01, epsilon=0.1, huberconst=0.5, method='obj'):
    '''This function provides a differentially-private estimate of the svm classifier according to Sarwate et al. 2011,
    "Differentially Private Empirical Risk Minimization" paper.

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

    if epsilon < 0.0:
        raise Exception('ERROR: Epsilon should be positive.')
    else:
        num = features.shape[0]  # number of samples
        dim = features.shape[1]  # dimension of a sample vector 
        # svm function only needs [vector x_i * label y_i], i.e., multiply all the training data vectors by their labels
        XY = features.multiply(labels[labels.columns[0]], axis="index")
        XY = XY.to_numpy(copy=False) 

        if method == 'obj':
            fpriv = train_svm_objectiveperturbation(XY, num, dim, epsilon, lambda_, huberconst)
        else:
            fpriv = train_svm_outputperturbation(XY, num, dim, epsilon, lambda_, huberconst)

        return fpriv
