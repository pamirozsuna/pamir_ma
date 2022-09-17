import numpy as np
import math
import pandas as pd

def covariance_mat(M):
    """ Returns the covariance matrix of rows of input matrix
    :parameter: np.ndarray or pd.DataFrame, shape n x p
    :return: n x p np.ndarray covariance matrix
    """
    if not isinstance(M, np.ndarray) and not isinstance(M, pd.DataFrame):           # Checks input type
        raise Exception('Input Matrix has to be np.array or pd.DataFrame')
    M=np.array(M)                               # Converts to np.array
    M=M-M.mean(axis=1).reshape(-1,1)            # Subtracts row means from each row
    C=np.matmul(M, M.T)/M.shape[1]             # Matrix multiplication yields the covariance matrix
    return C

def pearson_corr(M):
    """ Returns the pearson correlation coefficients between pairs of rows in M.
        The function uses matrix inversion to calculate the partial correlation.

        Parameters:
             M : array-like, shape (n, p)
            Array with the different variables. Each row of M is taken as a variable

        Returns:
            P : array-like, shape (n, n)
            P[i, j] contains the partial correlation of M[i, :] and M[j, :] controlling
            for the remaining variables in M.
        """
    M=np.asarray(M)
    n=M.shape[0]
    P=np.array([np.corrcoef((M[i,:], M[j,:]))[0,1] for i in range(n) for j in range(n)])
    return P.reshape(n,n)

def partial_corr_inv(M):
    """ Returns the partial linear correlation coefficients between pairs of rows in M.
    The function uses matrix inversion to calculate the partial correlation.

    Parameters:
         M : array-like, shape (n, p)
        Array with the different variables. Each row of M is taken as a variable

    Returns:
        P : array-like, shape (n, n)
        P[i, j] contains the partial correlation of M[i, :] and M[j, :] controlling
        for the remaining variables in M.
    """
    corr_M=pearson_corr(M)
    assert np.all(np.linalg.eigvals(corr_M) > 0), "Matrix is not positive definite."    #check if correlation Matrix is positive definite
    inv_M=np.linalg.inv(corr_M)
    P=np.array([-(inv_M[i,j])/math.sqrt(inv_M[i,i]*inv_M[j,j]) for i in range(inv_M.shape[0]) for j in range(inv_M.shape[0])])
    P=P.reshape(inv_M.shape)        #reshaping the 1x(p^2) dimensional array to a pxp dimensional array
    np.fill_diagonal(P, 1)          #fill diagonal with ones and return matrix
    return P

def partial_corr(M):
    """ Returns the partial linear correlation coefficients between pairs of rows in M.
    The function uses matrix inversion to calculate the partial correlation.

    Parameters:
         M : array-like, shape (n, p)
        Array with the different variables. Each row of M is taken as a variable

    Returns:
        P : array-like, shape (n, n)
        P[i, j] contains the partial correlation of M[i, :] and M[j, :] controlling
        for the remaining variables in M.
    """
    M=np.asarray(M).T
    part_corr = np.eye(M.shape[1], dtype=np.float)
    M=np.concatenate((M,np.ones((M.shape[0],1))), 1)
    for i in range(M.shape[1]-1):
        for j in range(i+1, M.shape[1]-1):
            idx=np.ones(M.shape[1], dtype=np.bool)
            idx[[i,j]]=False
            beta_i = np.linalg.lstsq(M[:, idx], M[:, i], rcond=None)[0]
            beta_j = np.linalg.lstsq(M[:, idx], M[:, j], rcond=None)[0]

            res_i = M[:, i] - M[:, idx].dot(beta_i)
            res_j = M[:,j] - M[:, idx].dot(beta_j)

            corr = np.corrcoef(res_i, res_j)[0,1]
            part_corr[i,j]=corr
            part_corr[j,i]=corr
    return part_corr

def dynmfc(TimeCourseBold, windowsize=30, stepsize=10, TR=2, method=1, output_windows=False):
    """Calculate the dynamical functional connectivity using a moving time window.
    :param: 1) Time course matrix of Bold Signal
            2) Windowsize in seconds
            3) Stepsize in seconds
            4) TR in seconds
            5) Method:  1 -- pearson correlation
                        2 -- partial correlation
            6) Output_windows : boolean variable defining if correlation matrices for each timewindow are outputet
    :return: ndim array containing the correlations between the region-wise correlation matrices for each time window
    """
    assert int(stepsize) >= int(TR), "Stepsize has to be greater than repetition time"
    tc=np.asarray(TimeCourseBold)
    nregions=tc.shape[0]
    wsize=np.around(float(windowsize)/float(TR))    #converts windowsize seconds to frames of Bold-Signal
    stpsize=np.around(float(stepsize)/float(TR))    #converts stepsize to Bold-Signal frames
    frnum=tc.shape[1]   #counts number of Bold-Signal frames
    ls_corr_mat=[]      #initiate list of correlation matrices
    strtpoint=0
    while strtpoint <= frnum-wsize:
        endpoint=strtpoint+int(wsize)                       #endpoint of current slice
        tc_slice=tc[:, strtpoint:endpoint]
        if method == 1:
            ls_corr_mat.append(pearson_corr(tc_slice))      #appends a nregions x nregions dimensional corrmatrix to list
        elif method == 2:
            ls_corr_mat.append(partial_corr_inv(tc_slice))  #appends the matrix using partial correlation
        strtpoint += int(stpsize)
    excl_fr=frnum-endpoint    #calculating the number of frames not included in the analysis.
    if excl_fr != 0:
        print(excl_fr,'Bold-frames not included in analysis.')

    if output_windows: return np.array(ls_corr_mat)         #outputs corrlation matrix for each time window

    fcd=np.empty((len(ls_corr_mat), len(ls_corr_mat)))      #initialize fcd matrix
    for i in range(len(ls_corr_mat)):
        for j in range(len(ls_corr_mat)):
            fcd[i,j]=np.corrcoef(np.array(ls_corr_mat[i]).reshape(1,nregions**2), np.array(ls_corr_mat[j]).reshape(1,nregions**2))[0,1]
    return fcd

