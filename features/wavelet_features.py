import numpy as np

def bounded_variation(coefficients, axis):
    '''Calculates the bounded variation using the derivatives of the coefficients.

        Parameters
        ----------
        coefficients : numpy array
             one-dimensional array with discrete fourier coefficients.
        Returns
        -------
        bounded_variation : float
             The final output is the bounded_variation for the discrete fourier coefficients.
    '''
    diffs = np.diff(coefficients, axis=axis)
    abs_sums = np.sum(np.abs(diffs), axis=axis)
    max_c = np.max(coefficients, axis=axis)
    min_c = np.min(coefficients, axis=axis)
    return np.divide(abs_sums, max_c - min_c)

