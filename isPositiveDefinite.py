import numpy as np


def isPositiveDefinite(A,tol,semi,verbose=True):
    result = True
    x = np.amin(np.linalg.eig((A))[0].real)
    if semi == False:
        if np.isnan(x) and verbose == True:
            print("Could not make sure that matrix is positive definite (eig threw NaN)!")
        else:
            if x<= 0 - tol:
                result = False
            elif x<= 0 and verbose == True:
                print("Matrix might not be completely positive definite, but within tolerance (-> numerical accuracy); smallest calculated eigenvalue: ")
                print(x)
    else:
        if np.isnan(x) and verbose == True:
            print("Could not make sure that matrix is positive semidefinite (eig threw NaN)!")
        else:
            if x < 0 - tol:
                result = False
            elif x < 0 and verbose == True:
                print("phs:isPositiveDefiniteHermitian:semiDefinitenessUnsure',['Matrix might not be completely positive semidefinite (computed eigenvalue is smaller than zero), but within tolerance (-> numerical accuracy); smallest calculated eigenvalue: "+ x)
    if result == False and verbose == True:
        print("Calculated eigenvalue is ")
        print(x)
    return result
