import numpy as np
from scipy.sparse import csr_matrix
from scipy import signal
from inputValidation import inputValidation
import matplotlib.pyplot as plt


class PHS:
    def __init__(self, J, R, Q, B,E=None, P=None, S=None, N=None, dim=1, inputValidation=True,verbose= True,inputTolerance = 1e-10):
            self.J=J      #SKEW-SYMMETRIC matrix describing the interconnection structure of the system
            self.R=R      #POSITIVE SEMI-DEFINITE SYMMETRIC matrix describing the dissipative behaviour of the system
            self.Q=Q      #POSITIVE SEMI-DEFINITE, SYMMETRIC matrix describing the energy storage behaviour of the system. (For descriptor-systems, restrictions differ slightly. -> see 'E')
            self.B=B      #input/output matrix describing the input/output-behaviour of the system.
            self.Jmor = csr_matrix((J), dtype = np.float)
            self.Jmor.eliminate_zeros()    
            self.Rmor = csr_matrix((R), dtype = np.float)
            self.Rmor.eliminate_zeros()
            self.Qmor = csr_matrix((Q), dtype = np.float)
            self.Qmor.eliminate_zeros()
            self.Bmor = csr_matrix((B), dtype = np.float)
            self.Bmor.eliminate_zeros()
            self.E = E
            self.Emor = csr_matrix((self.E), dtype = np.float)
            self.Emor.eliminate_zeros()            
            self.P = P
            self.Pmor = csr_matrix((self.P), dtype = np.float)
            self.Pmor.eliminate_zeros()            
            self.S = S
            self.Smor = csr_matrix((self.S), dtype = np.float)
            self.Smor.eliminate_zeros()
            self.N = N
            self.Nmor = csr_matrix((self.N), dtype = np.float) 
            self.Nmor.eliminate_zeros()
            #########################
            # self.E = np.identity(len(J))
            # self.Emor = csr_matrix((self.E), dtype = np.float)
            # self.Emor.eliminate_zeros()            
            # s = self.B.shape
            # self.P = np.zeros(s)
            # self.Pmor = csr_matrix((self.P), dtype = np.float)
            # self.S = np.zeros((s[1],s[1]))
            # self.Smor = csr_matrix((self.S), dtype = np.float)
            # self.N = np.zeros((s[1],s[1]))
            # self.Nmor = csr_matrix((self.N), dtype = np.float)            
            self.inputValidation = inputValidation 
            self.verbose = verbose 
            self.inputTolerance = inputTolerance
            self.flag_descriptor = False
            self.flag_MIMO = False
            self.flag_DAE = False
            self.dim = dim
            self.sys()

    
    def updateFlags(self, changedPropertyKey):
        if changedPropertyKey == "E":
            if self.E == np.identity(self.dim):
                self.flag_descriptor = False 
                self.flag_DAE = False
            else:
                self.flag_descriptor = True
                if np.linalg.matrix_rank(self.E)<self.Emor.shape[1]:#Check the numerics of shape &Emor
                    self.flag_DAE = True
                else:
                    self.flag_DAE = False
        elif changedPropertyKey == "B":
            if self.B.shape[1] > 1:#check numerics & dims 
                self.flag_MIMO = True
            else:
                self.flag_MIMO = False

    def parseInputs(self):#,J,R,Q,B,E=None, P=None, S=None, N=None):
        # self.J = J
        # self.R = R
        # self.Q = Q
        # self.B = B
        # self.E = E
        # self.P = P 
        # self.S = S
        # self.N = N
        s = self.B.shape

        if hasattr(self.E, "__len__"):
            if (self.E==None).all():
                self.E = np.identity(self.J.shape[0])
        else:
            if self.E==None:
                self.E = np.identity(self.J.shape[0])

        if hasattr(self.P, "__len__"):
            if (self.P==None).all():
                self.P = np.zeros((s[0],s[1]),float)
        else:
            if self.P==None:
                self.P = np.zeros((s[0],s[1]),float)

        if hasattr(self.S, "__len__"):
            if (self.S==None).all():
                self.S = np.zeros((s[1],s[1]),float)
        else:
            if self.S==None:
                self.S = np.zeros((s[1],s[1]),float)

        if hasattr(self.N, "__len__"):
            if (self.N==None).all():
                self.N = np.zeros((s[1],s[1]),float)
        else:
            if self.N==None:
                self.N = np.zeros((s[1],s[1]),float)

        

    def sys(self):
        self.parseInputs()#self.J,self.R,self.Q,self.B,self.E,self.P,self.S,self.N)
        # self.J = self.Jmor
        # self.R = self.Rmor
        # self.Q = self.Qmor
        # self.B = self.Bmor
        # self.E = self.Emor
        # self.P = self.Pmor
        # self.S = self.Smor
        # self.N = self.Nmor   
        if self.inputValidation == True:
            inputValidation(self)
        else:
            if self.verbose == True:
                print("No input validation")

    def get_J(self):
        return self._J
    def set_J(self, x):
        if np.all(x!=0) and self.verbose==True:
            print("phs:phs:ChangedProperty',..Property has been changed successfully.\nConsider running phs.inputValidation(sys) to validate new properties")
        self.J = x
    def get_R(self):
        return self._R
    def set_R(self, x):
        if np.all(x!=0) and self.verbose==True:
            print("phs:phs:ChangedProperty',..Property has been changed successfully.\nConsider running phs.inputValidation(sys) to validate new properties")
        self._R = x
    def get_Q(self):
        return self._Q
    def set_Q(self, x):
        if np.all(x!=0) and self.verbose==True:
            print("phs:phs:ChangedProperty',..Property has been changed successfully.\nConsider running phs.inputValidation(sys) to validate new properties")
        self._Q = x
    def get_B(self):
        return self._B
    def set_B(self, x):
        if np.all(x!=0) and self.verbose==True:
            print("phs:phs:ChangedProperty',..Property has been changed successfully.\nConsider running phs.inputValidation(sys) to validate new properties")
        self._B = x
        self.updateFlags("B")
    def get_E(self):
        return self._E
    def set_E(self, x):
        if np.all(x!=0) and self.verbose==True:
            print("phs:phs:ChangedProperty',..Property has been changed successfully.\nConsider running phs.inputValidation(sys) to validate new properties")
        self._E = x
        self.updateFlags("E")
    def get_P(self):
        return self._P
    def set_P(self, x):
        if np.all(x!=0) and self.verbose==True:
            print("phs:phs:ChangedProperty',..Property has been changed successfully.\nConsider running phs.inputValidation(sys) to validate new properties")
        self._P = x
    def get_S(self):
        return self._S
    def set_S(self, x):
        if np.all(x!=0) and self.verbose==True:
            print("phs:phs:ChangedProperty',..Property has been changed successfully.\nConsider running phs.inputValidation(sys) to validate new properties")
        self._S = x
    def get_N(self):
        return self._N
    def set_N(self, x):
        if np.all(x!=0) and self.verbose==True:
            print("phs:phs:ChangedProperty',..Property has been changed successfully.\nConsider running phs.inputValidation(sys) to validate new properties")
        self._N = x
    def get_dim(self):
        return self._dim
    def set_dim(self, x):
        if np.all(x!=0) and self.verbose==True:
            print("phs:phs:ChangedProperty',..Property has been changed successfully.\nConsider running phs.inputValidation(sys) to validate new properties")
        self._dim = x

    def phs2ss(self):#checking if it has bode plot & norm & step response PBSN
        if(self.flag_descriptor == False):
                ss = signal.StateSpace((self.Jmor-self.Rmor)*self.Qmor.todense(),(self.B-self.P),((self.B+self.P).transpose()*self.Qmor),(self.S+self.N))
        else:
            if (self.verbose == True):
                L = np.linalg.cholesky(self.E)
                # ss = signal.StateSpace(np.dot(np.dot(np.linalg.inv(L),(self.Jmor-self.Rmor)),self.Qmor*np.linalg.inv(L).transpose()),np.dot(np.linalg.inv(L),(self.B-self.P)),np.dot((self.B+self.P).transpose(),np.dot(self.Qmor,np.linalg.inv(L).transpose())),(self.S+self.N))
                ss = signal.StateSpace(np.dot(np.dot(np.linalg.inv(L),(self.Jmor-self.Rmor).todense()),np.dot(self.Qmor.todense(),np.linalg.inv(L).transpose())),np.dot(np.linalg.inv(L),(self.B-self.P)),np.dot((self.B+self.P).transpose(),self.Q*np.linalg.inv(L).transpose()),(self.S+self.N))

        return ss

    def makeFull(self):
        self.J = self.J.todense()
        self.R = self.R.todense()
        self.Q = self.Q.todense()
        self.B = self.B.todense()
        self.E = self.E.todense()
        self.P = self.P.todense()
        self.S = self.S.todense()
        self.N = self.N.todense()


def main():
    #n=2
    #L = np.array([[1, 2, 3, 4, 5]])
    #C = np.array([[1, 2, 3, 4, 5]])
    #Res = np.array([[1, 2, 3, 4, 5]])
    #J, R, Q, B = setup_LadderNetworkSystem(n,L,C,Res)
    #mag,phase,omega = control.bode(Gp)
    
    J = np.array([[0.,1.,0.,0.,0.,0.],[-1.,0.,1.,0.,0.,0.],[0.,-1.,0.,1.,0.,0.],[0.,0.,-1.,0.,1.,0.],[0.,0.,0.,-1.,0.,1.],[0.,0.,0.,0.,-1.,0.]])
    R = np.array([[0.,0.,0.,0.,0.,0.],[0.,0.5,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.5,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.]])
    Q = np.array([[1.0,0.,0.,0.,0.,0.],[0.,1.0,0.,0.,0.,0.],[0.,0.,0.5,0.,0.,0.],[0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.3333,0.],[0.,0.,0.,0.,0.,1.]])
    B = np.array([[1.0],[0.],[0.],[0.],[0.],[0.]])
    
    ############
    #OPTIONAL
    #E = np.identity(6)
    #P = np.zeros((6,6), float)
    #S = np.zeros((6,6), float)
    #N = np.zeros((6,6), float)
    ############

    A = PHS(J,R,Q,B)
    print(A.phs2ss())    
    w, mag, phase = signal.bode(A.phs2ss())

    fig, axs = plt.subplots(2)
    axs[0].set_title('Bode plot')

    axs[0].plot(w,mag)
    axs[0].grid()
    axs[0].set(xlabel='Freq', ylabel='Magnitude')
    axs[0].set_xscale("log")

    axs[1].plot(w,phase)
    axs[1].grid()
    axs[1].set(xlabel='Freq', ylabel='Phase')
    axs[1].set_xscale("log")
    plt.subplots_adjust(bottom=0.2, right=1, top=1.5)
    

if __name__ == "__main__":
    main()
