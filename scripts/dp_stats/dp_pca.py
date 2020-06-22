def dp_pca_ag(data, epsilon=1.0, delta=0.1):
    """
    This function provides a differentially-private estimate using Analyze Gauss method
    of the second moment matrix of the data

    Input:

      data = data matrix, samples are in columns
      epsilon = privacy parameter, defaul
      hat_A = (\epsilon, \delta)-differentially-private estimate of A = data*data'

    Example:

      >>> import numpy as np
      >>> import dp_stats as dps
      >>> data = np.random.rand(10)
      >>> hat_A = dps.dp_pca_ag ( data, 1.0, 0.1 )
      [[ 1.54704321  2.58597112  1.05587101  0.97735922  0.03357301]
       [ 2.58597112  4.86708836  1.90975259  1.41030773  0.06620355]
       [ 1.05587101  1.90975259  1.45824498 -0.12231379 -0.83844168]
       [ 0.97735922  1.41030773 -0.12231379  1.47130207  0.91925544]
       [ 0.03357301  0.06620355 -0.83844168  0.91925544  1.06881321]]

    """

    import numpy as np

    if any(np.diag(np.dot(data.transpose(), data))) > 1:
        print(
            "ERROR: Each column in the data matrix should have 2-norm bounded in [0,1]."
        )
        return
    elif epsilon < 0.0:
        print("ERROR: Epsilon should be positive.")
        return
    elif delta < 0.0 or delta > 1.0:
        print("ERROR: Delta should be bounded in [0,1].")
        return
    else:

        A = np.dot(data, data.transpose())
        D = (1.0 / epsilon) * np.sqrt(2.0 * np.log(1.25 / delta))
        m = len(A)
        temp = np.random.normal(0, D, (m, m))
        temp2 = np.triu(temp)
        temp3 = temp2.transpose()
        temp4 = np.tril(temp3, -1)
        E = temp2 + temp4
        hat_A = A + E
        return hat_A


def dp_pca_sn(data, epsilon=1.0):
    """
    This function provides a differentially-private estimate using Symmetric Noise method
    of the second moment matrix of the data

    Input:

      data = data matrix, samples are in columns
      epsilon = privacy parameter, default 1.0

    Output:

      hat_A = (\epsilon, \delta)-differentially-private estimate of A = data*data'

    Example:

      >>> import numpy as np
      >>> import dp_stats as dps
      >>> data = np.random.rand(10)
      >>> hat_A = dps.dp_pca_sn ( data, 1.0 )
      [[ 1.54704321  2.58597112  1.05587101  0.97735922  0.03357301]
       [ 2.58597112  4.86708836  1.90975259  1.41030773  0.06620355]
       [ 1.05587101  1.90975259  1.45824498 -0.12231379 -0.83844168]
       [ 0.97735922  1.41030773 -0.12231379  1.47130207  0.91925544]
       [ 0.03357301  0.06620355 -0.83844168  0.91925544  1.06881321]]

    """

    import numpy as np

    if any(np.diag(np.dot(data.transpose(), data))) > 1:
        print(
            "ERROR: Each column in the data matrix should have 2-norm bounded in [0,1]."
        )
        return
    elif epsilon < 0.0:
        print("ERROR: Epsilon should be positive.")
        return
    else:
        A = np.dot(data, data.transpose())
        d = len(A)
        nsamples = d + 1
        sigma = 1.0 / (2.0 * epsilon)
        Z_mean = 0.0
        Z = np.random.normal(Z_mean, sigma, (d, nsamples))
        E = np.dot(Z, Z.transpose())
        hat_A = A + E
        return hat_A


def dp_pca_ppm(data, k, Xinit, epsilon=1.0, delta=0.1):
    """
    This function provides a differentially-private estimate using Private Power method
    of the second moment matrix of the data

    Input:

      data = data matrix, samples are in columns
      k = reduced dimension
      Xinit = d x k size, initialization for the sampling
      epsilon = privacy parameter, default 1.0
      delta = privacy parameter, default 0.1

    Output:

      X = (\epsilon, \delta)-differentially-private estimate of the top-k subspace of A = data*data'

    Example:

      >>> import numpy as np
      >>> import dp_stats as dps
      >>> data = np.random.rand(10)
      >>> hat_A = dps.dp_pca_ppm ( data, 1.0, 0.1 )
      [[ 1.54704321  2.58597112  1.05587101  0.97735922  0.03357301]
       [ 2.58597112  4.86708836  1.90975259  1.41030773  0.06620355]
       [ 1.05587101  1.90975259  1.45824498 -0.12231379 -0.83844168]
       [ 0.97735922  1.41030773 -0.12231379  1.47130207  0.91925544]
       [ 0.03357301  0.06620355 -0.83844168  0.91925544  1.06881321]]

    """

    import numpy as np

    if any(np.diag(np.dot(data.transpose(), data))) > 1:
        print(
            "ERROR: Each column in the data matrix should have 2-norm bounded in [0,1]."
        )
        return
    elif epsilon < 0.0:
        print("ERROR: Epsilon should be positive.")
        return
    elif delta < 0.0 or delta > 1.0:
        print("ERROR: Delta should be bounded in [0,1].")
        return
    else:
        A = np.dot(data, data.transpose())
        d = np.size(A, 0)
        U, S, V = np.linalg.svd(A)
        param = S[k - 1] * np.log(d) / (S[k - 1] - S[k])
        L = round(10 * param)

        sigma = (1.0 / epsilon) * np.sqrt(4.0 * k * L * np.log(1.0 / delta))
        x_old = Xinit
        count = 0
        while count <= L:
            G_new = np.random.normal(
                0, np.linalg.norm(x_old, np.inf) * sigma, (d, k)
            )
            Y = np.dot(A, x_old) + G_new
            count += 1
            q, r = np.linalg.qr(Y)
            x_old = q[:, 0:k]
        X = x_old

        return X
