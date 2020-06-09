import numpy as np
#from statistics import stdev
# from pylab import norm
from scipy.optimize import minimize
from dp_func import noisevector


def lrloss(z): 
    hz = np.log(1 + np.exp(-z))
    return hz 

def eval_lr(weights, XY, num, lambda_, b):
        
    # add logistic loss from all samples ( 1/n * sum( huberloss(y_i w' x_i) ) )
    # fw = average( lrloss(XY.dot(weights)) )
    # ---------- XY = XY.dot(weights)
    # ---------- fw = np.mean(lrloss(XY.to_numpy()))
    XY = np.matmul(XY, weights)
    fw = np.mean(lrloss(XY))

    # add regularization term (1/2 * lambda * |w|^2) and b term (1/n * b'w) 
    fw += 0.5 * lambda_ * weights.dot(weights) + 1.0 / num * b.dot(weights) 
    return fw

def train_lr_nonpriv(XY, num, dim, lambda_):
    # return a non-private lr classifier

    w0 = np.zeros(dim)  # w starting point              ?? dataframe
    b = np.zeros(dim)  # zero noise vector
    res = minimize(eval_lr, w0, args=(XY, num, lambda_, b), 
                 method='L-BFGS-B', bounds=None)

    print("non-priv:")
    print("    w: ", res.x)
    print("    status: ", res.success)

    # if not res.success:
    #     raise Exception(res.message)
    f = res.x
    return f
    # return res.x, res.success

def train_lr_outputperturbation(XY, num, dim, lambda_, epsilon):
    # train a non-private lr classifier, then add noise
    # chaudhuri2011differentially: Algorithm 1 ERM with output perturbation
    
    f = train_lr_nonpriv(XY, num, dim, lambda_)

    beta = num * lambda_ * epsilon / 2
    noise = noisevector(dim, beta)
    fpriv = f + noise

    print("output:")
    print("    w: ", fpriv)

    return fpriv

def train_lr_objectiveperturbation(XY, num, dim, lambda_, epsilon):
    # chaudhuri2011differentially: Algorithm 2 ERM with objective perturbation

    c = 0.25  # value for lr
    tmp = c / (num * lambda_)
    epsilon_p = epsilon - np.log(1.0 + 2 * tmp + tmp * tmp)
    if epsilon_p < 1e-4:
        # raise Exception("Error: Cannot run algorithm for this lambda, epsilon and huberconst value")
        print("Error: Cannot run algorithm for this lambda, epsilon and huberconst value")
    
    w0 = np.zeros(dim)
    beta = epsilon_p / 2
    b = noisevector(dim, beta)
    res = minimize(eval_lr, w0, args=(XY, num, lambda_, b),
                   method='L-BFGS-B', bounds=None)

    print("objective:")
    print("    w: ", res.x)
    print("    status: ", res.success)

    # if not res.success:
    #     raise Exception(res.message)
    fpriv = res.x
    return fpriv   

def dp_lr(features, labels, lambda_=0.01, epsilon=0.1, method='obj'):
    '''
    This function provides a differentially-private estimate of the lr classifier 
    according to Sarwate et al. 2011, "Differentially Private Empirical Risk Minimization" paper.
    '''

    if epsilon < 0.0:
        # raise Exception('ERROR: Epsilon should be positive.')
        print("ERROR: Epsilon should be positive.")
    else:
        num = features.shape[0]  # number of samples
        dim = features.shape[1]  # dimension of a sample vector 
        # lr function only needs [vector x_i * label y_i], i.e., multiply all the training data vectors by their labels
        XY = features.multiply(labels[labels.columns[0]], axis="index") 
        XY = XY.to_numpy(copy=False)

        if method == 'obj':
            fpriv = train_lr_objectiveperturbation(XY, num, dim, epsilon, lambda_)
        else:
            fpriv = train_lr_outputperturbation(XY, num, dim, epsilon, lambda_)
 
        return fpriv
