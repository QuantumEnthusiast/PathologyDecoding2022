from scipy.stats import kurtosis as kurt
from scipy.stats import skew as skew
import numpy as np

def compute_svd_entropy(epochs, axis, **kwargs):
    def svd_entropy_1d(X, Tau, DE, W):
        '''Computes the entropy of the singular values taken from applying singular value decomposition.
            (see build_embedding)
        Parameters
        ----------
        a : numpy array
            one-dimensional floating-point array representing a time series.
        Tau: int
            the lag or delay when building embedding sequence
        D: the embedding dimension

        Returns
        -------
            FI : numpy array
            The final output is an array with values for the entropy for each singular value.
        '''

        if W is None:
            Y = build_embedding(X, Tau, DE)
            W = np.linalg.svd(Y, compute_uv=0)
            W /= sum(W)  # normalize singular values

        return -1 * sum(W * np.log(W))

    Tau = kwargs["Tau"]
    DE = kwargs["DE"]
    W = kwargs["W"]
    return np.apply_along_axis(svd_entropy_1d, axis, epochs, Tau, DE, W)

def compute_fisher_information(epochs, axis, **kwargs):
    def fisher_1d(a, tau, de):
        '''Computes the Fisher information of a single signal with embedding dimension "de" and delay "tau"
        (see build_embedding)

        Parameters
        ----------
        a : numpy array
            one-dimensional floating-point array representing a time series.
        Tau: int
            the lag or delay when building embedding sequence
        D: the embedding dimension

        Returns
        -------
        FI : float
             The final output is the Fisher information.
        '''
        matrix = build_embedding(a, tau, de)
        W = np.linalg.svd(matrix, compute_uv=False)
        W /= sum(W)  # normalize singular values
        FI_v = (W[1:] - W[:-1]) ** 2 / W[:-1]
        return np.sum(FI_v)

    tau = kwargs["Tau"]
    de = kwargs["DE"]
    return np.apply_along_axis(fisher_1d, axis, epochs, tau, de)

def compute_petrosian_fractal_dimension(epochs, axis):
    def pfd_1d(X, D=None):
        '''Compute Petrosian Fractal Dimension of a time series
        (see build_embedding)
        Parameters
        ----------
        X : list
            one-dimensional list of time series values.
        D: list (optional)
            the first order differential sequence of X

        Returns
        -------
        FI : numpy array
             The final output is an array with values for the Petrosian Fractal Dimension.
        '''

        N_delta = 0
        if D is None:
            D = np.diff(X)
            D = D.tolist()

        for i in range(1, len(D)):
            if D[i] * D[i - 1] < 0:
                N_delta += 1
        n = len(X)
        return np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * N_delta))
    return np.apply_along_axis(pfd_1d, axis, epochs)

def compute_higuchi_fractal_dimension(epochs, axis, **kwargs):
    def hfd_1d(X, Kmax):
        # taken from pyeeg
        """ Compute Hjorth Fractal Dimension of a time series X, kmax
         is an HFD parameter
        """
        L = []
        x = []
        N = len(X)
        for k in range(1, Kmax):
            Lk = []
            for m in range(0, k):
                Lmk = 0
                for i in range(1, int(np.floor((N - m) / k))):
                    Lmk += abs(X[m + i * k] - X[m + i * k - k])
                Lmk = Lmk * (N - 1) /np.floor((N - m) / float(k)) / k
                Lk.append(Lmk)
            L.append(np.log(np.mean(Lk)))
            x.append([np.log(float(1) / k), 1])

        (p, r1, r2, s) = np.linalg.lstsq(x, L, rcond=None)
        return p[0]
    Kmax = kwargs["Kmax"]
    return np.apply_along_axis(hfd_1d, axis, epochs, Kmax)

def compute_hurst_exponent(epochs, axis):
    def hurst_1d(X):
        '''Compute the Hurst exponent of X. If the output H=0.5,the behavior
            of the time-series is similar to random walk. If H<0.5, the time-series
            cover less "distance" than a random walk, vice verse.

           Parameters
           ----------
           X : list
               list containing time series values.

           Returns
           -------
           H : float
                Hurst exponent.
           '''

        X = np.array(X)
        N = X.size
        T = np.arange(1, N + 1)
        Y = np.cumsum(X)
        Ave_T = Y / T

        S_T = np.zeros(N)
        R_T = np.zeros(N)
        for i in range(N):
            S_T[i] = np.std(X[:i + 1])
            X_T = Y - T * Ave_T[i]
            R_T[i] = np.ptp(X_T[:i + 1])

        for i in range(1, len(S_T)):
            if np.diff(S_T)[i - 1] != 0:
                break
        for j in range(1, len(R_T)):
            if np.diff(R_T)[j - 1] != 0:
                break
        k = max(i, j)
        assert k < 10, "Error, reconsider k < 10"

        R_S = R_T[k:] / S_T[k:]
        R_S = np.log(R_S)

        n = np.log(T)[k:]
        A = np.column_stack((n, np.ones(n.size)))
        [m, c] = np.linalg.lstsq(A, R_S, rcond=None)[0]
        H = m
        return H
    return np.apply_along_axis(hurst_1d, axis, epochs)


def build_embedding(X, Tau, D):
    '''Build a set of embedding sequences from given time series X with lag Tau
    and embedding dimension DE. Let X = [x(1), x(2), ... , x(N)], then for each
    i such that 1 < i <  N - (D - 1) * Tau, we build an embedding sequence,
    Y(i) = [x(i), x(i + Tau), ... , x(i + (D - 1) * Tau)]. All embedding
    sequence are placed in a matrix Y.

    Parameters
    ----------
    X : list
        list containing time series values.
    Tau: int
        the lag or delay when building embedding sequence
    D: the embedding dimension

    Returns
    -------
    Y : list
         The final output is a 2-D list of the embedding matrix built.
    '''
    shape = (X.size - Tau * (D - 1), D)
    strides = (X.itemsize, Tau * X.itemsize)
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


def compute_fractal_dimension(epochs, axis):
    diff1 = np.diff(epochs)
    sum_of_distances = np.sum(np.sqrt(diff1 * diff1), axis=axis)
    max_dist = np.apply_along_axis(lambda epoch: np.max(np.sqrt(np.square(epoch - epoch[0]))), axis, epochs)
    return np.divide(np.log10(sum_of_distances), np.log10(max_dist))

def hjorth_activity(epochs, axis):
    return np.var(epochs, axis=axis)

def hjorth_complexity(epochs, axis):
    diff1 = np.diff(epochs, axis=axis)
    diff2 = np.diff(diff1, axis=axis)
    sigma1 = np.std(diff1, axis=axis)
    sigma2 = np.std(diff2, axis=axis)
    return np.divide(np.divide(sigma2, sigma1), hjorth_mobility(epochs, axis))

def hjorth_mobility(epochs, axis):
    diff = np.diff(epochs, axis=axis)
    sigma0 = np.std(epochs, axis=axis)
    sigma1 = np.std(diff, axis=axis)
    return np.divide(sigma1, sigma0)

def compute_hjorth_parameters(epochs, axis):
    activity = np.var(epochs, axis=axis)
    diff1 = np.diff(epochs, axis=axis)
    diff2 = np.diff(diff1, axis=axis)
    sigma0 = np.std(epochs, axis=axis)
    sigma1 = np.std(diff1, axis=axis)
    sigma2 = np.std(diff2, axis=axis)
    mobility = np.divide(sigma1, sigma0)
    complexity = np.divide(np.divide(sigma2, sigma1), hjorth_mobility(epochs, axis))
    return activity, complexity, mobility

def energy(epochs, axis):
    return np.mean(epochs*epochs, axis=axis)

def non_linear_energy(epochs, axis):
    return np.apply_along_axis(lambda epoch: np.mean((np.square(epoch[1:-1]) - epoch[2:] * epoch[:-2])), axis, epochs)

def zero_crossing(epochs, axis):
    e = 0.01
    norm = epochs - epochs.mean()
    return np.apply_along_axis(lambda epoch: np.sum((epoch[:-5] <= e) & (epoch[5:] > e)), axis, norm)

def zero_crossing_derivative(epochs, axis):
    e = 0.01
    diff = np.diff(epochs)
    norm = diff-diff.mean()
    return np.apply_along_axis(lambda epoch: np.sum(((epoch[:-5] <= e) & (epoch[5:] > e))), axis, norm)

def line_length(epochs, axis):
    return np.sum(np.abs(np.diff(epochs)), axis=axis)

def kurtosis(epochs, axis):
    return kurt(epochs, axis=axis, bias=False)

def skewness(epochs, axis):
    return skew(epochs, axis=axis, bias=False)

def line_length(epochs, axis):
    return np.sum(np.abs(np.diff(epochs)), axis=axis)

def maximum(epochs, axis):
    return np.max(epochs, axis=axis)

def mean(epochs, axis):
    return np.mean(epochs, axis=axis)

def median(epochs, axis):
    return np.median(epochs, axis=axis)

def minimum(epochs, axis):
    return np.min(epochs, axis=axis)

