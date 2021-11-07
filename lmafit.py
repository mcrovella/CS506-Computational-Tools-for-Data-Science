import numpy as np
# translated from lmafit_mc_adp.m by Mark Crovella December 2014
# omitting many options and just implementing the core functionality
# for documentation on lmafit see http://lmafit.blogs.rice.edu/
#
# Note: this is very useful:
# http://wiki.scipy.org/NumPy_for_Matlab_Users
#
def lmafit_mc_adp(m,n,k,Known,data,opts=None):
    """
 Output:
           X --- m x k matrix
           Y --- k x n matrix
         Out --- output information
 Input:
        m, n --- matrix sizes
           k --- rank estimate
       Known is a 2xL ndarray holding indices of known elements 
        data --- values of known elements in a 1D row vector
        opts --- option structure (not used)
    """
    L = len(data)
    tol = 1.25e-4
    maxit = 500
    iprint = 0
    reschg_tol = 0.5*tol
    datanrm = np.max([1.0,np.linalg.norm(data)])
    objv = np.zeros(maxit)
    RR = np.ones(maxit)
    if iprint == 1:
        print('Iteration: ')
    if iprint == 2:
        print('\nLMafit_mc: \n')

    # initialize: make sure the correctness of the index set and data
    #data[data==0]=np.spacing(1)
    data_tran = False
    Z = np.zeros((m,n))
    Z[Known] = data

    if m>n:
        tmp = m
        m = n
        n = tmp
        Z = Z.T
        Known = np.nonzero(Z)
        data = Z[Known]
        data_tran = True

    #no inital solution
    if opts is None:
        X = np.zeros((m,k))
        Y = np.zeros((k,n))
    #with inital solution
    else:
        X = opts['U']
        Y = opts['V']
        if data_tran:
            tX = X
            X = Y.T
            Y = tX.T

    Res = data
    res = datanrm

    # parameters for alf
    alf = 0
    increment = 1
    itr_rank = 0
    minitr_reduce_rank = 5
    maxitr_reduce_rank = 50

    for iter in range(maxit):
        itr_rank += 1
        Xo = X
        Yo = Y
        Res0 = Res
        res0 = res
        alf0x = alf
        # iterative step
        # Zfull option only
        Zo = Z
        X = Z.dot(Y.T)
        X, R = np.linalg.qr(X)
        Y = X.T.dot(Z)
        Z = X.dot(Y)
        Res = data - Z[Known]
        #
        res = np.linalg.norm(Res)
        relres = res/datanrm
        ratio = res/res0
        reschg = np.abs(1-res/res0)
        RR[iter] = ratio
        # omitting rank estimation
        # adjust alf
        if ratio>=1.0:
            increment = np.max([0.1*alf, 0.1*increment])
            X = Xo
            Y = Yo
            Res = Res0
            res = res0
            relres = res/datanrm
            alf = 0
            Z = Zo
        elif ratio>0.7:
            increment = max(increment, 0.25*alf)
            alf = alf+increment;
    
        if iprint==1:
            print('{}'.format(iter))
        if iprint==2:
            print('it: {} rk: (none), rel. {} r. {} chg: {} alf: {} inc: {}\n'.format(iter, k, relres,ratio,reschg,alf0x,increment))

        objv[iter] = relres

        # check stopping
        if ((reschg < reschg_tol) and ((itr_rank > minitr_reduce_rank) or (relres < tol))):
            break
    
        Z[Known] = data + alf*Res

    if iprint == 1:
        print('\n')

    if data_tran:
        tX = X
        X = Y.T
        Y = tX.T

    obj = objv[:iter]

    return X, Y, [obj, RR, iter, relres, reschg]
