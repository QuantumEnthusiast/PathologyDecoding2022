from scipy.signal import hilbert
import numpy as np

def instantaneous_phases(band_signals, axis):
    '''Computes the synchrony between the band signals using a phase shift to every frequency component.

    Parameters
    ----------
    band_signals : numpy array
         one-dimensional array including the signal values.
    Returns
    -------
    instantaneous_phase : float
         The final output is instantaneous_phases from the band signal.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
    .. [2] Alan V. Oppenheim, Ronald W. Schafer. Discrete-Time Signal Processing,
    ..     Third Edition, 2009. Chapter 12. ISBN 13: 978-1292-02572-8
    '''

    analytical_signal = hilbert(band_signals, axis=axis)
    return np.unwrap(np.angle(analytical_signal), axis=axis)


def phase_locking_values(inst_phases):
    ''' Computes the phase Locking Value (PLV) between two phases that can be used to investigate task-induced
      changes in long range synchronization of neural activity.

      Parameters
      ----------
      theta1 : float
           phase of signal 1.
      theta2 : float
           phase of signal 2.
      Returns
      -------
      phase_locking_value : numpy array
           The final output in an array of phase_locking_values between two phases.
      '''

    (n_windows, n_bands, n_signals, n_samples) = inst_phases.shape

    plvs = []
    for electrode_id1 in range(n_signals):
        # only compute upper triangle of the synchronicity matrix and fill
        # lower triangle with identical values
        for electrode_id2 in range(electrode_id1+1, n_signals):
            for band_id in range(n_bands):
                theta1 = inst_phases[:, band_id, electrode_id1]
                theta2 = inst_phases[:, band_id, electrode_id2]

                delta = np.subtract(theta1, theta2)
                xs_mean = np.mean(np.cos(delta), axis=-1)
                ys_mean = np.mean(np.sin(delta), axis=-1)
                plv = np.linalg.norm([xs_mean, ys_mean], axis=0)

                plvs.append(plv)

    # n_window x n_bands * (n_signals*(n_signals-1))/2
    plvs = np.array(plvs).T
    return plvs



