from scipy.io import savemat
from scipy.sparse import csr_matrix
import numpy as np

def matlab_interface(sys):
    Jsparsed = csr_matrix((sys.J), dtype = np.float)
    Rsparsed = csr_matrix((sys.R), dtype = np.float)
    Qsparsed = csr_matrix((sys.Q), dtype = np.float)
    Bsparsed = csr_matrix((sys.B), dtype = np.float)
    Esparsed = csr_matrix((sys.E),dtype = np.float)
    Psparsed = csr_matrix((sys.P),dtype = np.float)
    Ssparsed = csr_matrix((sys.S),dtype = np.float)
    Nsparsed = csr_matrix((sys.N),dtype = np.float)
    sparsed_matrices = {'J':Jsparsed,
                        'R':Rsparsed,
                        'Q':Qsparsed,
                        'B':Bsparsed,
                        'E':Esparsed,
                        'P':Psparsed,
                        'S':Ssparsed,
                        'N':Nsparsed,
                        'dim':sys.dim,
                        'flag_descriptor':sys.flag_descriptor,
                        'flag_MIMO':sys.flag_MIMO,
                        'flag_DAE':sys.flag_DAE,
                        'inputValidation':sys.inputValidation,
                        'verbose':sys.verbose,
                        'inputTolerance':sys.inputTolerance}
    
    savemat("sparsed_matrices.mat", sparsed_matrices)