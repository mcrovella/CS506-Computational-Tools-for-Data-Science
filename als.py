import numpy as np

def als(X,k,lambda_,max_iter,threshold):
    
    """
 Output:
           U --- n x k matrix
           V --- k x d matrix
 Input:
           X --- n x d input matrix
           k --- rank estimate
           Lambda --- Ridge regularizer parameter
           max_iter --- maximum number of iterations
           threshold --- stopping criterion(minimum improvement in RMSE) 
    """
    def solve_V(X,W,U):
        X = X.values
        n,d = X.shape
        V = np.zeros((d,k))
        X = X.T
        W = W.T.values
        I = lambda_*np.eye(k)
        for j,x_j in enumerate(X):
            v_j = np.linalg.solve(U[W[j]].T.dot(U[W[j]])+I, U[W[j]].T.dot(x_j[W[j]]))
            V[j] = v_j
        return V

    def solve_U(X,W,V):
        X = X.values
        W = W.values
        n,d = X.shape
        U = np.zeros((n,k))
        I = lambda_*np.eye(k)
        for i,x_i in enumerate(X):
            u_i = np.linalg.solve(V[W[i]].T.dot(V[W[i]])+I, V[W[i]].T.dot(x_i[W[i]]))
            U[i] = u_i
        return U

    W = ~X.isnull()
    n,d = X.shape
    U = np.ones((n,k))
    V = solve_V(X,W,U)
    n_known = float(W.sum().sum())
    RMSE = np.sqrt((X - U.dot(V.T)).pow(2).sum().sum()/n_known)
    RMSEs=[RMSE]
    for i in range(max_iter):
        U_new = solve_U(X,W,V)
        V_new = solve_V(X,W,U_new)
        RMSE_new = np.sqrt((X - U_new.dot(V_new.T)).pow(2).sum().sum()/n_known)
        if (RMSE - RMSE_new) < threshold:
            RMSEs.append(RMSE_new)
            break
        else:
            RMSEs.append(RMSE_new)
            RMSE = RMSE_new
            U = U_new
            V = V_new
    #print "Error history",RMSEs
    return U, V.T
