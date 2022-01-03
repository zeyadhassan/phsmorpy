import numpy as np
from scipy.sparse import coo_matrix, bmat
from isPositiveDefinite import isPositiveDefinite
from numpy import linalg as LA

def inputValidation(sys):
    #checking if dimentions are correct
    sJ = sys.J.shape
    sR = sys.R.shape
    sQ = sys.Q.shape
    sB = sys.B.shape
    sE = sys.E.shape
    sP = sys.P.shape
    sN = sys.N.shape
    sS = sys.S.shape
    if sJ[0] != sJ[1]:
        print("Error - wrong input : J not square")
    if sR[0] != sR[1] or sJ != sR:
        print("Error - wrong input : R not square or dimension different to J")
    if sQ[0] != sQ[1] or sJ != sQ:
        print("Error - wrong input : Q not square or dimension different to J")
    if sE[0] != sE[1] or sJ != sE:
        print("Error - wrong input : E not square or dimension different to J")
    if sB[0] != sJ[0]:
        print("Error - wrong input : B has wrong state dimension")
    if sB != sP:
        print("phsmor:phs:wrongInput: dimensions of P are not correct, different dimensions of P and B")
    if sN != sS:
        print("phsmor:phs:wrongInput: dimensions of S and N must be equal, dimensions of S and N not equal")
    if sN[0] != sB[1] or sN[1] != sB[1]:
        print("phsmor:phs:wrongInput: dimensions of S and N not correct (compared to B)")


    if ((sys.S.transpose() == sys.S)[0,0] == False):
        norm_S = LA.norm((sys.S-sys.S.transpose()).todense())/LA.norm((sys.S).todense())
        if norm_S > sys.inputTolerance:
            print("S is not symmetric:\n norm(S-S')/norm(S)")
    
    
    if(sys.N.transpose() == -sys.N)[0,0] == False:
        norm_N = LA.norm((sys.N+sys.N.transpose()).todense())/LA.norm((sys.N).todense())
        if norm_N > sys.inputTolerance:
            print("phsmor:phs:wrongInput")
    
    if(sys.Q.transpose()*sys.E == (sys.Q.transpose()*sys.E).transpose())[0,0] == False:
        norm_QE = LA.norm(((sys.Q.transpose()*sys.E - sys.E.transpose()*sys.Q)).todense())/LA.norm((sys.Q.transpose()*sys.E).todense())
        if norm_QE > sys.inputTolerance:
            print("Operator defined by E, Q and J is not skew-adjoint (Q'*E must be symmetric):\n norm(Q'*E - E'*Q)/norm(Q'*E) = ")
   

    #if (np.allclose(sys.Q.transpose(1, 0, 2), sys.Q) == False):
    if(-sys.Q.transpose()*sys.J*sys.Q == (sys.Q.transpose()*sys.J*sys.Q).transpose())[0,0] == False:
        norm_QJQ = LA.norm((sys.Q.transpose()*sys.J*sys.Q+sys.Q.transpose()*sys.J.transpose()*sys.Q))/LA.norm((sys.Q.transpose()*sys.J*sys.Q))
        if norm_QJQ > sys.inputTolerance:
            norm_J = LA.norm((sys.J+sys.J.transpose()))/LA.norm((sys.J))
            if norm_J < sys.inputTolerance:
                print("phsmor:phs:alternativeValidation","Passed input validation with skew-symmetry of J instead of Q'*J*Q")
            else:
                print("Operator defined by E, Q and J is not skew-adjoint (Q'*J*Q must be skew-symmetric):\n norm(Q'*J*Q+Q'*J'*Q)/norm(Q'*J*Q) = ")


    #########CHECK########### scipy bmat
    # A = coo_matrix([[1, 2], [3, 4]])
    #WW = np.concatenate(([sys.Q.transpose()*sys.R*sys.Q, sys.Q.transpose()*sys.P], [sys.P.transpose()*sys.Q, sys.S]), axis=0)
    #A = coo_matrix([[sys.Q.transpose()*sys.R*sys.Q],[sys.Q.transpose()*sys.P]])
    # B = coo_matrix([[5], [6]])
    #B = coo_matrix([[sys.P.transpose()*sys.Q],[sys.S]])
    #C = coo_matrix([sys.P.transpose()*sys.Q* sys.S])
    #WW = bmat([[A],[B]]).toarray()
    #print(WW)
    W = np.bmat([[(sys.Qmor.transpose()*sys.Rmor*sys.Qmor).todense(), sys.Qmor.transpose()*sys.P], [sys.P.transpose()*sys.Qmor, sys.S]])
    #print(W)
    if ((W.transpose() == W)[1,2]) == False:
        W_transp = np.bmat(([[(sys.Qmor.transpose()*sys.Rmor.transpose()*sys.Qmor).todense(), sys.Qmor.transpose()*sys.P], [sys.P.transpose()*sys.Qmor, sys.S.transpose()]]))
        norm_W = LA.norm(W-W_transp)/LA.norm(W)#
        if norm_W > sys.inputTolerance:
            print("W = [Q'*R*Q, Q'*P; P'*Q, S] is not symmetric!:\n norm(W-W')/norm(W) = ")

    if isPositiveDefinite(W,sys.inputTolerance, True, sys.verbose)==False:
        print("W = [Q'*R*Q, Q'*P; P'*Q, S] is not positive semi-definite")
        print("phsmor:phs:wrongInput")


#if (inputValidation == True):
#    self.inputValidation(sys)
#else:
#    if (verbose == True):
#        print("No input validation for correctness of port-Hamiltonian system!")
#self.makeSparse(sys)
  
