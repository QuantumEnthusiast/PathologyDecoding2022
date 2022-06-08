import numpy as np

def maximum(amplitude_spectrum, axis=-1):
    '''Computes the maximum value of the EEG signals.

    Parameters
    ----------
    amplitude_spectrum : numpy array
         Multidimensional array, where (rows = channels, columns = estimated fourier coefficients).
    Returns
    -------
    max : float
         The final output is the maximal value in de power spectrum.
    '''
    return np.max(amplitude_spectrum, axis=axis)

def mean(amplitude_spectrum, axis=-1):
    '''Computes the value mean of the EEG signals.

    Parameters
    ----------
    amplitude_spectrum : numpy array
         Multidimensional array, where (rows = channels, columns = estimated fourier coefficients).
    Returns
    -------
    max : float
         The final output is the mean value in de power spectrum.
    '''
    return np.mean(amplitude_spectrum, axis=axis)

def minimum(amplitude_spectrum, axis=-1):
    '''Computes the minimum value of the EEG signals.

    Parameters
    ----------
    amplitude_spectrum : numpy array
         Multidimensional array, where (rows = channels, columns = estimated fourier coefficients).
    Returns
    -------
    max : float
         The final output is the minimum value in de power spectrum.
    '''
    return np.min(amplitude_spectrum, axis=axis)

def peak_frequency(amplitude_spectrum, axis=-1):
    '''Computes the maximum peak frequency of the EEG signals.

    Parameters
    ----------
    amplitude_spectrum : numpy array
         Multidimensional array, where (rows = channels, columns = estimated fourier coefficients).
    Returns
    -------
    max : float
         The final output is the maximal peak frequency found in de power spectrum.
    '''
    return amplitude_spectrum.argmax(axis=axis)

def power(amplitude_spectrum, axis=-1):
    '''Computes the power spectrum the EEG signals.

    Parameters
    ----------
    amplitude_spectrum : numpy array
         Multidimensional array, where (rows = channels, columns = estimated fourier coefficients).
    Returns
    -------
    max : numpy array
         The final output is an array of discrete fourier coefficients that represent the power spectrum.
    '''
    return np.sum(amplitude_spectrum * amplitude_spectrum, axis=axis)

def power_ratio(powers, axis=-1):
    '''Computes the power ratio between the EEG signals.

    Parameters
    ----------
    powers : numpy array
         Multidimensional array, where (rows = channels, columns = estimated fourier coefficients).
    Returns
    -------
    max : numpy array
         The final output is an array of the power ratio between the estimated fourier coefficients.
    '''
    ratios = powers / np.sum(powers, axis=axis, keepdims=True)
    return ratios

def spectral_entropy(ratios, axis=None):
    '''Computes the spectral entropy from the power ratio.

    Parameters
    ----------
    ratios : numpy array
         Multidimensional array, where (rows = channels, columns = power-ratios).
    Returns
    -------
    max : float
         The final output is the maximal peak frequency found in de power spectrum.
    '''
    return -1 * ratios * np.log(ratios)

def variance(amplitude_spectrum, axis=-1):
    '''Computes the variance of the EEG signals.

    Parameters
    ----------
    amplitude_spectrum : numpy array
         Multidimensional array, where (rows = channels, columns = estimated fourier coefficients).
    Returns
    -------
    max : float
         The final output is the variance of the EEG signal.
    '''
    return np.var(amplitude_spectrum, axis=axis)

def value_range(amplitude_spectrum, axis=-1):
    return np.ptp(amplitude_spectrum, axis=axis)