import numpy as np
from phs_final import PHS

def setup_LadderNetworkSystem(n,L,C,Res):
    # self.n = n
    # self.L = L
    # self.C = C
    # self.Res = Res    
    J = None
    R = None
    Q = None
    B = None
    
    if np.mod(n,2) != 0:
        print('This system must have an even order');
    # J
    J = np.add(np.diag(np.ones(n-1), 1), np.diag(-1*np.ones(n-1), -1))

    # np.tile(L, (3,2))
    if Res.shape[1] == 1:
        Res = np.tile(Res,(int(n/2)+1,1))

    diagRIndex = (n,1)
    diagR = np.zeros(diagRIndex).transpose()
    index_Res = 0
    for index_diagR in range(1, n, 2):
        diagR[0,index_diagR] = Res[0, index_Res]
        index_Res += 1
    diagR[0,len(diagR.transpose())-1] = diagR[0,len(diagR.transpose())-1] + Res[0,len(Res.transpose())-1]
    R = np.diag(diagR[0])
        
    # Q
    diagQIndex = (n,1)    
    diagQ = np.zeros(diagQIndex).transpose()
    CInverse = np.reciprocal(C)
    LInverse = np.reciprocal(L)
    index_C = 0
    index_L = 0
    for index_diagQ_C in range(0,n-1,2):
        diagQ[0,index_diagQ_C] = CInverse[0,index_C]
        #diagQ[index_diagQ_C] = np.linalg.inv(C)
        index_C += 1
    for index_diagQ_L in range (1,n,2):
        diagQ[0,index_diagQ_L] = LInverse[0,index_L]
        index_L += 1
    Q = np.diag(diagQ[0]);
    

    #B
    BIndex = (n,1) 
    B = np.zeros(BIndex)
    B[0,0] = 1
    
    sys = PHS(J,R,Q,B)
    return sys
    #matlab_interface(sys)

def main():
    
    n=6
    
    L = np.array([[1.0,1.0,1.0]])#MUST BE FLOAT INPUT VALUES, ROW VECTOR .. Line 35, 40
    C = np.array([[1.0, 2.0, 3.0]])#MUST BE FLOAT INPUT VALUES, ROW VECTOR .. Line 36, 44
    R = np.array([[0.5, 0.5, 0.5, 0.5]])#MUST BE ROW VECTOR? Line 29
    AAA = setup_LadderNetworkSystem(n,L,C,R)
    


if __name__ == "__main__":
    main()
    