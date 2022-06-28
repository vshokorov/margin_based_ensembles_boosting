import numpy as np

def scatter_with_disp(xarr, yarr, label, ax, color=None, marker=None):
    yarr = np.array(yarr)
    xarr = np.array(xarr)
    
    res_mean = np.array([np.mean(yarr[xarr == x]) for x in xarr])
    res_error = np.array([np.std(yarr[xarr == x]) for x in xarr])
    
    index = np.argsort(xarr)
    yarr = yarr[index]
    res_mean = res_mean[index]
    res_error = res_error[index]
    xarr = xarr[index]

    ax.scatter(xarr, yarr, color, label = label, marker = marker)
    ax.plot(xarr, res_mean, '-', color='gray')
    ax.fill_between(xarr, res_mean - res_error, res_mean + res_error,
                    color='gray', alpha=0.2)