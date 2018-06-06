import numpy as np
#from statistics import stdev
# from pylab import norm
from scipy.optimize import minimize


def noisevector( scale, Length ):

    r1 = np.random.normal(0, 1, Length)#standard normal distribution
    n1 = np.linalg.norm( r1, 2 )#get the norm of this random vector
    r2 = r1 / n1#the norm of r2 is 1
    normn = np.random.gamma( Length, 1/scale, 1 )#Generate the norm of noise according to gamma distribution
    res = r2 * normn#get the result noise vector
    return res

def huber(z, h):#chaudhuri2011differentially corollary 21

    if z > 1 + h:
        hb = 0
    elif np.fabs(1-z) <= h:
        hb = (1 + h - z)**2 / (4 * h)
    else:
        hb = 1 - z
    return hb

def svm_output_train(data, labels, epsilon, Lambda, h):

    N = len( labels )
    l = len( data[0] )#length of a data point
    scale = N * Lambda * epsilon / 2
    noise = noisevector( scale, l )
    x0 = np.zeros(l)#starting point with same length as any data point

    def obj_func(x):
        jfd = huber( labels[0] * np.dot(data[0],x), h )
        for i in range(1,N):
            jfd = jfd + huber( labels[i] * np.dot(data[i],x), h )
        f = ( 1/N )*jfd + (1/2) * Lambda * ( np.linalg.norm(x,2)**2 )
        return f

    #minimization procedure
    f = minimize(obj_func, x0, method='Nelder-Mead').x #empirical risk minimization using scipy.optimize minimize function
    fpriv = f + noise
    return fpriv

def svm_objective_train(data, labels,  epsilon, Lambda, h):

    #parameters in objective perturbation method
    c = 1 / ( 2 * h )#chaudhuri2011differentially corollary 13
    N = len( labels )#number of data points in the data set
    l = len( data[0] )#length of a data point
    x0 = np.zeros(l)#starting point with same length as any data point
    Epsilonp = epsilon - 2 * np.log( 1 + c / (Lambda * N))
    if Epsilonp > 0:
        Delta = 0
    else:
        Delta = c / ( N * (np.exp(epsilon/4)-1) ) - Lambda
        Epsilonp = epsilon / 2
    noise = noisevector(Epsilonp/2, l)

    def obj_func(x):
        jfd = huber( labels[0] * np.dot(data[0], x), h)
        for i in range(1,N):
            jfd = jfd + huber( labels[i] * np.dot(data[i], x), h )
        f = (1/N) * jfd + (1/2) * Lambda * (np.linalg.norm(x,2)**2) + (1/N) * np.dot(noise,x) + (1/2)*Delta*(np.linalg.norm(x,2)**2)
        return f

    #minimization procedure
    fpriv = minimize(obj_func, x0, method='Nelder-Mead').x#empirical risk minimization using scipy.optimize minimize function
    return fpriv

def dp_svm(data, labels, method='obj', epsilon=0.1, Lambda = 0.01, h = 0.5):
    '''
    This function provides a differentially-private estimate of the svm classifier according to Sarwate et al. 2011,
    "Differentially Private Empirical Risk Minimization" paper.

    Input:

      data = data matrix, samples are in rows
      labels = labels of the data samples
      method = 'obj' (for objective perturbation) or 'out' (for output perturbation)
      epsilon = privacy parameter, default 1.0
      Lambda = regularization parameter
      h = huber loss parameter

    Output:

      fpriv = (\epsilon)-differentially-private estimate of the svm classifier

    Example:

      >>> import numpy as np
      >>> import dp_stats as dps

      >>> X = np.random.normal(1.0, 1.0, (n,d));
      >>> Y = np.random.normal(-1.0, 1.0, (n,d));
      >>> labelX = 1.0 * np.ones(n);
      >>> labelY = -1.0 * np.ones(n);

      >>> data = np.vstack((X,Y));
      >>> labels = np.hstack((labelX,labelY));

      >>> fpriv = dps.classification.dp_svm (data, labels, 'obj', 0.1, 0.01, 0.5)
      [  9.23418189   2.63380995  -2.01654661  -1.19112074  17.32083386
         3.37943017 -14.76815378  12.3119061   -1.82132988  24.03559848]

    '''

    import numpy as np

    if epsilon < 0.0:
        print('ERROR: Epsilon should be positive.')
        return
    else:
        if method == 'obj':
            fpriv = svm_objective_train(data, labels, epsilon, Lambda, h)
        else:
            fpriv = svm_output_train(data, labels, epsilon, Lambda, h)

        return fpriv