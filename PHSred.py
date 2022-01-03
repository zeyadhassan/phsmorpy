import numpy as np
from phs_final import PHS
from scipy import signal
import matplotlib.pyplot as plt

class PHSred(PHS):
    def __init__(self, J, R, Q, B, method,E=None, P=None, S=None, N=None, dim=1, inputValidation=True,verbose= True,inputTolerance = 1e-10):
        self.method = method
        PHS.__init__(self, J, R, Q, B, E, P, S, N, dim, inputValidation, verbose, inputTolerance)  

    def get_method(self):
        print('Reduction Method used is: ',self.method)

def main():
    
    J = np.array([[0.,1.,0.,0.,0.,0.],[-1.,0.,1.,0.,0.,0.],[0.,-1.,0.,1.,0.,0.],[0.,0.,-1.,0.,1.,0.],[0.,0.,0.,-1.,0.,1.],[0.,0.,0.,0.,-1.,0.]])
    R = np.array([[0.,0.,0.,0.,0.,0.],[0.,0.5,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.5,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.]])
    Q = np.array([[1.0,0.,0.,0.,0.,0.],[0.,1.0,0.,0.,0.,0.],[0.,0.,0.5,0.,0.,0.],[0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.3333,0.],[0.,0.,0.,0.,0.,1.]])
    B = np.array([[1.0],[0.],[0.],[0.],[0.],[0.]])
    
    A = PHSred(J,R,Q,B,'effort constraint')
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
    
    A.get_method()
    
if __name__ == "__main__":
    main()
