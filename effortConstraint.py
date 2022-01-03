import numpy as np
import scipy as sp
from phs_final import PHS
from PHSred import PHSred
from scipy.linalg import solve_continuous_lyapunov
import control.matlab as mt


def effortConstraint(sys, redOrder):
    if sys.flag_DAE == True:
        print("Error - effortConstraint method is not applicable to DAE systems!")

    if sys.flag_descriptor == True:
        print("Warning - Transforming descriptor system (implicit) to explicit system... Computation of matrix inverse is performed!")
        sys.Q = sys.Q/sys.E
        sys.E = np.identity(sys.E.shape[0])
    
    A = np.dot(sys.Q,(sys.J - sys.R))
    B = np.dot(sys.Q,sys.B) - np.dot(sys.Q,sys.P)
    C = sys.B.transpose() + sys.P.transpose()
    
    # if np.rank(A) !=  A.shape[1]:
    #     print("Error - phsmor:phsMOR:badInput', 'The system matrix A = Q*(J-R) does not have full rank!")
    #smallest negative eigenvalue+add to the diagonal an order of magnitude of the same value
    ##########
    Wc = solve_continuous_lyapunov(A, np.dot(-B,B.transpose()))
    Wo = solve_continuous_lyapunov(A.transpose(), np.dot(-C.transpose(),C))
    Wc = Wc + 1e-12*np.eye(A.shape[0])
    Wo = Wo + 1e-12*np.eye(A.shape[0])
    S = np.linalg.cholesky(Wc).transpose()
    R = np.linalg.cholesky(Wo).transpose()
    S = S.transpose()
    R = R.transpose()
    # S = mt.lyap(A,B)	
    # R = mt.lyap(A.transpose(),C.transpose())	
    U, SIGMA, V = sp.linalg.svd(np.dot(S,R.transpose()))
    V = V.transpose()
    T = np.dot(np.dot(np.diag(SIGMA**(-0.5)),V.transpose()),R)
    T_inv = np.dot(S.transpose(),np.dot(U,np.diag(SIGMA**(-0.5))))

    J_bal = np.dot(T_inv.transpose(),np.dot(sys.J,T_inv))
    R_bal = np.dot(T_inv.transpose(),np.dot(sys.R,T_inv))
    Q_bal = np.dot(T,np.dot(sys.Q,T.transpose()))
    B_bal = np.dot(T_inv.transpose(),sys.B)
    P_bal = np.dot(T_inv.transpose(),sys.P)
    
    J_red = J_bal[0:redOrder,0:redOrder]
    R_red = R_bal[0:redOrder,0:redOrder]
    Q_help = np.linalg.solve(Q_bal[redOrder::,redOrder::],Q_bal[redOrder::,0:redOrder])
    Q_red = Q_bal[0:redOrder,0:redOrder] - np.dot(Q_bal[0:redOrder, redOrder::],Q_help)
    B_red = B_bal[0:redOrder, :]
    P_red = P_bal[0:redOrder, :]
    
    # create reduced system
    redSys = PHS(J_red, R_red, Q_red, B_red, np.identity(Q_red.shape[0]), P_red, sys.S, sys.N,'effort constraint');
    
    return redSys

