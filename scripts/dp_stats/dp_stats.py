def dp_mean(data_vec, epsilon=1.0, delta=0.1):
    """
    This function provides a differentially-private estimate of the mean of a vector.

    Input:

      data_vec = data bounded in the range [0,1], use array instead of list
      epsilon = privacy parameter, default 1.0
      delta = privacy parameter, default 0.1

    Output:

      a scalar.


    Example:

      >>> import numpy as np
      >>> import dp_stats as dps
      >>> x = np.random.rand(10)
      >>> x_mu = dps.dp_mean( x, 1.0, 0.1 )
      [ 0.57438844]

    """

    import numpy as np

    if len(data_vec) != np.size(data_vec):
        print('ERROR: Input should be a vector.')
        return
    elif ( any(data_vec < 0.0) or any(data_vec > 1.0) ):
        print('ERROR: Input vector should have bounded entries in [0,1].')
        return
    elif epsilon < 0.0:
        print('ERROR: Epsilon should be positive.')
        return
    elif delta < 0.0 or delta > 1.0:
        print('ERROR: Delta should be bounded in [0,1].')
        return
    else:

        n = len(data_vec)
        f = np.mean(data_vec)
        if delta == 0:
            noise = np.random.laplace(loc = 0, scale = 1/float(n*epsilon), size = (1,1))
        else:
            sigma = (1.0/(n*epsilon))*np.sqrt(2*np.log(1.25/delta))
            noise = np.random.normal(0.0, sigma, 1)
        f += noise

        return f
        
def dp_var( data_vector,epsilon=1.0,delta=0.1 ):
    """
    This function provides a differentially-private estimate of the variance of a vector.

    Input:

      data_vector = data bounded in the range [0,1], use array instead of list
      epsilon = privacy parameter, default 1.0
      delta = privacy parameter, default 0.1

    Output:

      a scalar.

    Example:

      >>> import numpy as np
      >>> import dp_stats as dps
      >>> x = np.random.rand(10)
      >>> x_mu = dps.dp_var( x, 1.0, 0.1 )
      [ 0.37882534]

    """

    import numpy as np

    if len(data_vector) != np.size(data_vector):
        print('ERROR: Input should be a vector.')
        return
    elif any(data_vector < 0.0) or any(data_vector > 1.0):
        print('ERROR: Input vector should have bounded entries in [0,1].')
        return
    elif epsilon < 0.0:
        print('ERROR: Epsilon should be positive.')
        return
    elif delta < 0.0 or delta > 1.0:
        print('ERROR: Delta should be bounded in [0,1].')
        return
    else:
        n = len(data_vector)
        mu = np.mean(data_vector)
        var = (1.0/n) * sum((value - mu) ** 2 for value in data_vector)
        delf = 3.0 * (1.0-1.0/n)/n

        if delta == 0:
            noise = np.random.laplace(loc = 0, scale = delf/epsilon, size = (1,1))
        else:
            sigma = (3.0/(n*epsilon))*(1-1.0/n)*np.sqrt(2*np.log(1.25/delta))
            noise = np.random.normal(0.0, sigma, 1)

        var += noise

        return var
        
def dp_hist ( data, num_bins=10, epsilon=1.0, delta=0.1, histtype = 'continuous' ):
    """
    This function provides a differentially-private estimate of the histogram of a vector.

    Input:

      data = data vector
      num_bins = number of bins for the histogram, default is 10
      epsilon = privacy parameter, default 1.0
      delta = privacy parameter, default 0.1
      histtype = a string indicating which type of histogram is desired ('continuous', or 'discrete'),
                 by default, histtype = 'continuous'

    Note that for discrete histogram, the user input "num_bins" is ignored.

    Output:

      dp_hist_counts = number of items in each bins
      bin_edges = location of bin edges

    Example:

      >>> import numpy as np
      >>> import dp_stats as dps
      >>> x = np.random.rand(10)
      >>> x_mu = dps.dp_hist( x, 5, 1.0, 0.1, 'continuous' )
      (array([ 0.81163273, -1.18836727, -1.18836727,  0.81163273, -0.18836727]), array([ 0.1832111 ,  0.33342489,  0.48363868,  0.63385247,  0.78406625,
    0.93428004]))

    """

    import numpy as np

    if len(data) != np.size(data):
        print('ERROR: Input should be a vector.')
        return
    elif type(num_bins) != int:
        print('ERROR: Number of bins should be an integer')
        return
    elif epsilon < 0.0:
        print('ERROR: Epsilon should be positive.')
        return
    elif delta < 0.0 or delta > 1.0:
        print('ERROR: Delta should be bounded in [0,1].')
        return
    else:
        if histtype == 'discrete':
            num_bins = len( np.unique(data) )
        hist_counts = [0] * num_bins
        data_min = min(data)
        data_max = max(data)
        bin_edges = np.linspace(data_min, data_max, num_bins+1)
        interval = (data_max - data_min) + 0.000000000001
        
        for kk in data:
            loc = (kk - data_min) / interval
            index = int(loc * num_bins)
            hist_counts[index] += 1.0

        if delta==0:
            noise = np.random.laplace(loc = 0, scale = 1.0/epsilon, size = (1,len(hist_counts)))
        else:
            sigma = (1.0/epsilon)*np.sqrt(2*np.log(1.25/delta))
            noise = np.random.normal(0.0, sigma, len(hist_counts))

        hist_array=np.asarray(hist_counts)
        noise_array=np.asarray(noise)
        dp_hist_counts = hist_array+noise_array

        return ( dp_hist_counts, bin_edges )
