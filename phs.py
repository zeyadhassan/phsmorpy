import numpy as np
import scipy as sps
from scipy.sparse import csr_matrix
from scipy import signal
import matplotlib.pyplot as plt
#import control
import warnings
warnings.filterwarnings('ignore')
#import setup_LadderNetworkSystem
#import sys
#sys.path.append('src\mor\helpers')
#from nx import read_plot_output


############################INPUT_VALIDATION#####################################
def inputValidation(sys):
    #checking if dimentions are correct
    if sys.J.shape[0] != sys.J.shape[1]:
        print("Error - wrong input : dimensions of J are not correct")
    if sys.R.shape[0] != sys.R.shape[1] or np.equal(sys.J, sys.R):
        print("Error - wrong input : dimensions of R are not correct")
    if sys.Q.shape[0] != sys.Q.shape[1] or np.equal(sys.J, sys.Q):
        print("Error - wrong input : dimensions of R are not correct")
    if sys.B.shape[0] != sys.B.shape[1] or np.equal(sys.J, sys.B):
        print("Error - wrong input : dimensions of R are not correct")

    def isPositiveDefinite(A,tol,semi,Opts_verbose):
        result = True
        x = np.amin(np.linalg.eig(np.full(A))[0].real)
        if semi == False:
            if np.isnan(x) and Opts_verbose == True:
                print("Could not make sure that matrix is positive definite (eig threw NaN)!")
            else:
                if x<= 0 - tol:
                    result = False
                elif x<= 0 and Opts_verbose == True:
                    print("Matrix might not be completely positive definite, but within tolerance (-> numerical accuracy); smallest calculated eigenvalue: " + x)
        else:
            if np.isnan(x) and Opts_verbose == True:
                print("Could not make sure that matrix is positive semidefinite (eig threw NaN)!")
            else:
                if x < 0 - tol:
                    result = False
                elif x < 0 and Opts_verbose == True:
                    print("phs:isPositiveDefiniteHermitian:semiDefinitenessUnsure',['Matrix might not be completely positive semidefinite (computed eigenvalue is smaller than zero), but within tolerance (-> numerical accuracy); smallest calculated eigenvalue: "+ x)
        if result == False and Opts_verbose == True:
            print("Calculated eigenvalue is " + x)
        return result

    #def is_symmetric(A, tol=1e-8):
    #    return scipy.sparse.linalg.norm(A-A.T, scipy.Inf) < tol;    

    if (sys.S.transpose() == sys.S).all()==False:
        norm_S = np.norm(np.full(np.subtract(sys.S,sys.S.transpose())))/np.norm(np.full(sys.S));
        if norm_S > sys.Opts_inputTolerance:
            print("S is not symmetric:\n norm(S-S')/norm(S)")
    
    
    if(sys.N.transpose() == -sys.N).all() == False:
        norm_N = np.norm(np.full(np.add(sys.N,sys.N.transpose())))/np.norm(np.full(sys.N))
        if norm_N > sys.Opts_inputTolerance:
            print("phsmor:phs:wrongInput")

    #sys.Q[np.isnan(sys.Q)] = 0
    #if (np.allclose(sys.Q.transpose(1, 0, 2), sys.Q) == False):
    if(np.multiply(sys.Q.transpose(),sys.E) == np.multiply(sys.E.transpose(),sys.Q)).all() == False:
        norm_QE = np.norm(np.full(np.substract(np.multiply(sys.Q.transpose()),sys.E), np.multiply(sys.E.transpose(),sys.Q)))/np.norm(np.full(np.multiply(sys.Q.transpose(),sys.E)));
        if norm_QE > sys.Opts_inputTolerance:
            print("Operator defined by E, Q and J is not skew-adjoint (Q'*E must be symmetric):\n norm(Q'*E - E'*Q)/norm(Q'*E) = " + norm_QE)
   

    #if (np.allclose(sys.Q.transpose(1, 0, 2), sys.Q) == False):
    if(np.multiply(sys.Q.transpose(),sys.J,sys.Q) == -np.multiply(sys.Q.transpose(),sys.J,sys.Q)).all() == False:            
        norm_QJQ = np.norm(np.full(np.add(np.multiply(sys.Q.transpose(),sys.J,sys.Q),np.multiply(sys.Q.transpose(),sys.J.transpose(),sys.Q))))/np.norm(np.full(np.multiply(sys.Q.transpose(),sys.J,sys.Q)))
        if norm_QJQ > sys.Opts_inputTolerance:
            norm_J = np.norm(np.full(np.add(sys.J,sys.J.transpose())))/np.norm(np.full(sys.J))
            if norm_J < sys.Opts_inputTolerance:
                print("phsmor:phs:alternativeValidation","Passed input validation with skew-symmetry of J instead of Q'*J*Q")
            else:
                print("Operator defined by E, Q and J is not skew-adjoint (Q'*J*Q must be skew-symmetric):\n norm(Q'*J*Q+Q'*J'*Q)/norm(Q'*J*Q) = "+ norm_QJQ)

#def __init__(self, J, R, Q, B, dim=np.identity(1)):
    if isPositiveDefinite(np.multiply(sys.Q.transpose(),sys.E),sys.Opts_inputTolerance, 1, sys.Opts_verbose)==False:
        print("Q'*E is not positive semi-definite (Hamiltonian function must be bounded from below!)")
        print("phsmor:phs:wrongInput")
    
    W = np.array(np.full([[np.multiply(sys.Q.transpose(),sys.R,sys.Q), np.multiply(sys.Q.transpose(),sys.P)], [np.multiply(sys.P.transpose(),sys.Q), sys.S]]))
    if (W.transpose() == W).all() == False:
        W_transp = np.array[[np.multiply(sys.Q.transpose(),sys.R.transpose(),sys.Q), np.multiply(sys.Q.transpose(),sys.P)], [np.multiply(sys.P.transpose(),sys.Q), sys.S.transpose()]]
        norm_W = np.norm(np.substract(W,W_transp))/np.norm(W)
        if norm_W > sys.Opts_inputTolerance:
            print("W = [Q'*R*Q, Q'*P; P'*Q, S] is not symmetric!:\n norm(W-W')/norm(W) = " + norm_W)

    if isPositiveDefinite(W,sys.Opts_inputTolerance, 1, sys.Opts_verbose)==False:
        print("W = [Q'*R*Q, Q'*P; P'*Q, S] is not positive semi-definite")
        print("phsmor:phs:wrongInput")
#if (Opts_inputValidation == True):
#    self.inputValidation(sys)
#else:
#    if (Opts_verbose == True):
#        print("No input validation for correctness of port-Hamiltonian system!")
#self.makeSparse(sys)
  


class PHS:

    def __init__(self, J, R, Q, B, dim=np.identity(1)):
        #Opts = dict({
        #    "inputValidation": True,
        #    "verbose": True,
        #    "inputTolerance": 1e-10,
        #}.items())
        self.J=J      #SKEW-SYMMETRIC matrix describing the interconnection structure of the system
        self.R=R      #POSITIVE SEMI-DEFINITE SYMMETRIC matrix describing the dissipative behaviour of the system
        self.Q=Q      #POSITIVE SEMI-DEFINITE, SYMMETRIC matrix describing the energy storage behaviour of the system. (For descriptor-systems, restrictions differ slightly. -> see 'E')
        self.B=B      #input/output matrix describing the input/output-behaviour of the system.
        self.Opts_inputValidation = Opts_inputValidation = True
        self.Opts_verbose = Opts_verbose = True
        self.Opts_inputTolerance = Opts_inputTolerance = 1e-10
        self.__flag_descriptor
        self.__flag_MIMO
        self.__flag_DAE
        self.dim

    def updateFlags(self, changedPropertyKey):#
        if changedPropertyKey == "E":
            if self.E == np.identity(self.dim):
                self.__flag_descriptor = False 
                self.__flag_DAE = False
            else:
                self.__flag_descriptor = True
                if self.E.sps.sparse.csr_matrix.todense():#missing????
                    self.__flag_DAE = True
                else:
                    self.__flag_DAE = False
        elif changedPropertyKey == "B":
            if self.B.shape[2] > 1:
                self.__flag_MIMO = True
            else:
                self.__flag_MIMO = False


    def sys(self, J, R, Q, B):
        
        
        def parseInputs(J,R,Q,B):
            s = B.shape
            self.E = np.identity(len(J))
            self.P = np.zeros(s)
            self.S = np.zeros(s[2],s[2])
            self.N = np.zeros(s[2],s[2])
    

        parseInputs(J,R,Q,B)
        
        import pdb
        pdb.set_trace()
        J = sps.sparse.csr_matrix((J), dtype = np.float)
        J.sps.sparse.csr_matrix.eliminate_zeros()
        R = sps.sparse.csr_matrix((R), dtype = np.float)
        R.sps.sparse.csr_matrix.eliminate_zeros()
        Q = sps.sparse.csr_matrix((Q), dtype = np.float)
        Q.sps.sparse.csr_matrix.eliminate_zeros()
        B = sps.sparse.csr_matrix((B), dtype = np.float)
        B.sps.sparse.csr_matrix.eliminate_zeros()

        if self.Opts_inputValidation == True:
            inputValidation(self)
        else:
            if self.Opts_verbose == True:
                print("No input validation")
        

    def get_J(self):
        return self._J
    def set_J(self, x):
        if np.all(x!=0) and self.Opts_verbose==True:
            print("phs:phs:ChangedProperty',..Property has been changed successfully.\nConsider running phs.inputValidation(sys) to validate new properties")
        self.J = x
    def get_R(self):
        return self._R
    def set_R(self, x):
        if np.all(x!=0) and self.Opts_verbose==True:
            print("phs:phs:ChangedProperty',..Property has been changed successfully.\nConsider running phs.inputValidation(sys) to validate new properties")
        self._R = x
    def get_Q(self):
        return self._Q
    def set_Q(self, x):
        if np.all(x!=0) and self.Opts_verbose==True:
            print("phs:phs:ChangedProperty',..Property has been changed successfully.\nConsider running phs.inputValidation(sys) to validate new properties")
        self._Q = x
    def get_B(self):
        return self._J
    def set_B(self, x):
        if np.all(x!=0) and self.Opts_verbose==True:
            print("phs:phs:ChangedProperty',..Property has been changed successfully.\nConsider running phs.inputValidation(sys) to validate new properties")
        self._B = x
        self.updateFlags("B")
    def get_dim(self):
        return self._dim
    def set_dim(self, x):
        if np.all(x!=0) and self.Opts_verbose==True:
            print("phs:phs:ChangedProperty',..Property has been changed successfully.\nConsider running phs.inputValidation(sys) to validate new properties")
        self._dim = x
    def getMatrices(self):
        return self.sys

    #Operator Overloading
    #def minus()
    #def mtimes()
    #def plus()

    def phs2ss(sys):#checking if it has bode plot & norm & step response
        if(sys.__flag_descriptor == False):
            ss = signal.StateSpace(np.multiply(np.subtract(sys.J, sys.R),sys.Q), np.subtract(sys.B,sys.P), np.multiply((np.add(sys.B,sys.P)).transpose(),sys.Q), np.add(sys.S,sys.N))
        else:
            if (sys.Opts_verbose == True):
                #ss = signal.StateSpace(np.multiply(np.subtract(sys.J, sys.R),sys.Q), np.subtract(sys.B,sys.P), np.multiply((np.add(sys.B,sys.P)).transpose(),sys.Q), np.add(sys.S,sys.N))
                return ss
 
    #def phs2sss(sys):
    #    sys = sss(np.multiply(np.subtract(sys.J, sys.R),sys.Q), np.subtract(sys.B,sys.P), np.multiply(np.add(sys.B,sys.P).transpose(),sys.Q), np.add(sys.S,sys.N), sys.E)

    #def sss(sys):
    #    phs2sss(sys)

    def makeFull(sys):
        sys.J = np.full(sys.J);
        sys.R = np.full(sys.R);
        sys.Q = np.full(sys.Q);
        sys.B = np.full(sys.B);

    def makeSparse(sys):
        sys.J = csr_matrix((sys.J), dtype = np.float)
        sys.J.csr_matrix.eliminate_zeros()
        sys.R = csr_matrix((sys.R), dtype = np.float)
        sys.R.csr_matrix.eliminate_zeros()
        sys.Q = csr_matrix((sys.Q), dtype = np.float)
        sys.Q.csr_matrix.eliminate_zeros()
        sys.B = csr_matrix((sys.B), dtype = np.float)
        sys.B.csr_matrix.eliminate_zeros()
        return sys
        
        
def setup_LadderNetworkSystem(n,L,C,Res):
    if np.mod(n,2) != 0:
        print('This system must have an even order')
    # J
    Jindex = (n-1,1)
    J = np.add(np.diag(np.ones(Jindex), 1), np.diag(-1*np.ones(Jindex), -1))

    # R  a.shape[1] or size(a, axis=1) - repmat(a,2,3)
    if Res.shape[1] == 1:
        Res = np.kron(np.ones((n/2+1,1)),Res)
    
    diagRIndex = (n,1)    
    diagR = np.zeros(diagRIndex)
    index_Res = 1
    R = None
    for index_diagR in range(2, n, 2):
        diagR[index_diagR] = Res[index_Res]
        index_Res += 1
        diagR[len(diagR)] = np.add(diagR(len(diagR)), Res(len(Res)))
        R = np.diag(diagR)
        
    # Q
    diagQIndex = (n,1)    
    diagQ = np.zeros(diagQIndex);
    for index_diagQ_C in range(1,n-1,2):
        diagQ[index_diagQ_C] = np.linalg.inv(C)
    for index_diagQ_L in range (2,n,2):
        diagQ[index_diagQ_L] = np.linalg.inv(L)
    Q = np.diag(diagQ)
    #B
    BIndex = (n,1) 
    B = np.zeros(BIndex)
    B[1] = 1
    return J, R, Q, B

def main():
    n=2
    L = np.array([[1, 2, 3, 4, 5]])
    C = np.array([[1, 2, 3, 4, 5]])
    Res = np.array([[1, 2, 3, 4, 5]])
    J, R, Q, B = setup_LadderNetworkSystem(n,L,C,Res)
    #mag,phase,omega = control.bode(Gp)
    print(J)
    print(R)
    print(Q)
    print(B)    
    
    
if __name__ == "__main__":
    main()


