import numpy as np
from phs_final import PHS
from effortConstraint import effortConstraint
from scipy import signal
import matplotlib.pyplot as plt 
import time
from matlab_interface import matlab_interface
from setup_LadderNetworkSystem import setup_LadderNetworkSystem

def main():
    
    #n = 6
    #L = np.array([[1.0,1.0,1.0]])#MUST BE FLOAT INPUT VALUES, ROW VECTOR .. Line 35, 40
    #C = np.array([[1.0, 2.0, 3.0]])#MUST BE FLOAT INPUT VALUES, ROW VECTOR .. Line 36, 44
    #R = np.array([[0.5, 0.5, 0.5, 0.5]])#MUST BE ROW VECTOR? Line 29
    size = 50
    n=size*2
    L = np.ones((1,size))
    C = 2*np.ones((1,size))
    R = np.ones((1,(1+size)))
    A = setup_LadderNetworkSystem(n,L,C,R)

    start = time.time()

    # J = np.array([[0.,1.,0.,0.,0.,0.],[-1.,0.,1.,0.,0.,0.],[0.,-1.,0.,1.,0.,0.],[0.,0.,-1.,0.,1.,0.],[0.,0.,0.,-1.,0.,1.],[0.,0.,0.,0.,-1.,0.]])
    # R = np.array([[0.,0.,0.,0.,0.,0.],[0.,0.5,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.5,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.]])
    # Q = np.array([[1.0,0.,0.,0.,0.,0.],[0.,1.0,0.,0.,0.,0.],[0.,0.,0.5,0.,0.,0.],[0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.3333,0.],[0.,0.,0.,0.,0.,1.]])
    # B = np.array([[1.0],[0.],[0.],[0.],[0.],[0.]])
    ###########################
    #OPTIONAL
    #E = np.identity(6)
    #P = np.zeros((6,6), float)
    #S = np.zeros((6,6), float)
    #N = np.zeros((6,6), float)
    ###########################
    # A = PHS(J,R,Q,B)
    
    print(A.phs2ss())
    w, mag, phase = signal.bode(A.phs2ss())
    
    #####PLOTTING - FOM #####    
    fig, axs = plt.subplots(2)
    axs[0].set_title('Bode plot')
    axs[0].plot( w,mag, label='FOM')
    axs[1].plot( w,phase, label='FOM')
    ##########################

    redSys = effortConstraint(A, 2)
    w, mag, phase = signal.bode(redSys.phs2ss())

    #####PLOTTING - ROM #####
    axs[0].plot(w,mag,  '--', label='ROM')
    axs[0].grid()
    axs[0].set(xlabel='Freq', ylabel='Magnitude')
    axs[0].legend()
    axs[1].plot(w,phase, '--', label='ROM')
    axs[1].grid()
    axs[1].set(xlabel='Freq', ylabel='Phase')
    axs[1].legend()
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    plt.subplots_adjust(bottom=0.2, right=1, top=1.5)
    plt.show()
    #########################
    
    matlab_interface(A)
    
    print("Time for calculations:")
    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()