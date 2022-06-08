import numpy as np
from scipy.signal import coherence, welch
import config as c

def WelchEstimate(x, fs, NOVERLAP, NFFT, WINDOW, fmin, fmax):
    
    '''Estimates the power spectral density using Welch's method.
       Welch's method computes an estimate of the power spectral density by dividing the data into overlapping segments, 
       computing a modified periodogram for each segment and averaging the periodograms. 
        
    Parameters
    ----------
    x : numpy array
         1D array of the channel values.
         Input array requires either common reference or laplacian montage.
         
    fs : int
         Sampling frequency.
    NOVERLAP : int
         Number of points to overlap between segments. If None, noverlap = nperseg // 2.
    NFFT : int
         Length of the FFT used, if a zero padded FFT is desired.
    WINDOW : int 
        Length of each segment.
    fmin : int
        Minimum frequency bound for the power spectrum.
    fmax : int
        Maximum frequency bound for the power spectrum.
    
    Returns
    -------
    f : numpy array
        Array containing the frequency values of the power spectrum.
    Pxx : numpy array 
        Array containing Welch's coefficients bounded to fmin and fmax.

    References
    ----------
    .. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
    '''
    
    # Compute Welch's discrete fourier coefficients
    f, Pxx = welch(x, fs=fs, window='hann', 
                    noverlap=NOVERLAP, 
                    nfft=NFFT,
                    nperseg=WINDOW) 
    
    # Bound the frequency spectrum
    f_lim = f[(f>=fmin) & (f<=fmax)]           
    start = f.tolist().index(f_lim[0])      # get first freq index
    end = f.tolist().index(f_lim[-1])+1     # get last freq index
    Pxx = Pxx[start:end]                    # bound between [fmin,fmax]
    
    return f, Pxx


def QAPG(EEG_closed, channels):
    
    '''Computes the anterior to posterior ratio (gradient) of the Alpha power.
        Rhythmic activity from a normal wake brain should be distributed
        with an anterio-posterior gradient over the scalp: higher
        frequency beta activity with low voltages more prominently over
        the frontal regions fading posteriorly, and slower waves (e.g., alpha
        and mu rhythm) with higher voltages over the parietal and occipital lobes.
        
    Parameters
    ----------
    EEG_closed : numpy array
         Multidimensional array, where (rows = channels, columns = timepoints).
         Input array requires a re-referenced laplacian montage, using eyes closed states only.
    
    channels : numpy array
         1D array with the ordered channel names.

    Returns
    -------
    Qapg : float
         The final output is a quantified and normalised value for the anterio-posterior gradient.
        
        If Qapg < 0.4: Alpha power gradient is categorized as normal. 
        If 0.4 < Qapg < 0.6: Alpha power gradient is considered moderately differentiated.
        If Qapg > 0.6: Alpha power gradient is considered abnormal.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007
    '''
    
    assert len(EEG_closed) and len(channels) == 19, "Exception: Expected 19 channels"
    assert len(EEG_closed) == len(channels), "Exception: Returned Shape Mismatch"
    assert type(EEG_closed) and type(channels) == np.ndarray, "Exception: Returned Type Mismatch"
    assert EEG_closed.shape[1] >= 250*5, "Exception: To few data-points recorded"

    fmin = 8
    fmax = 12

    anterior = ['fp1','fp2','f7','f8','f3','fz','f4']
    posterior = ['t5','t6','p3','p4','pz','o1','o2']

    discrete_power_coefs_anterior = []
    discrete_power_coefs_posterior = []

    # Compute Welch's coefficients for each anterior and posterior channels
    for x in range(len(channels)):
        for i in range(len(anterior)):
            if anterior[i] in channels[x]:
                _, Pxx = WelchEstimate(EEG_closed[x], c.fs, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
                discrete_power_coefs_anterior.append(Pxx)

        for i in range(len(posterior)):
            if posterior[i] in channels[x]:
                _, Pxx = WelchEstimate(EEG_closed[x], c.fs, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
                discrete_power_coefs_posterior.append(Pxx)

    # Sum the total power for Pant and Ppos
    Pant = np.array(discrete_power_coefs_anterior).sum(axis=0).sum()
    Ppos = np.array(discrete_power_coefs_posterior).sum(axis=0).sum()

    # Compute the normalized anterio–posterior ratio
    Qapg = Pant / (Pant + Ppos)
    
    return float(Qapg)


def QSLOW(EEG_closed):
    
    '''Computes diffuse slow-wave activity of the EEG.
       Diffused slowing results in an increased power over the Delta [0-4]Hz and Theta [4-8]Hz bands  
       and decreased power in the Alpha [8-12]Hz and Beta bands [13-25]Hz.
       
       Using the guideline above, the mean spectrum of the EEG is calculated and the power ratio between
       Plow = {2. . .8} Hz and Pwide = {2. . .25} Hz
        
    Parameters
    ----------
    EEG_closed : numpy array
         Multidimensional array, where (rows = channels, columns = timepoints).
         Input array requires a common reference montage, using eyes closed states only.

    Returns
    -------
    Qslow : float
         The final output is a quantified and normalised value for diffused slow-wave activity.
        
        If Qslow > 0.6: EEG is categorized as abnormal. 
        If Qslow < 0.6: EEG is categorized as normal.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007
    '''
    
    assert len(EEG_closed) == 19, "Exception: Expected 19 channels"
    assert EEG_closed.shape[1] >= 250*5, "Exception: To few data-points recorded"

    fmin = 2
    fmax_wide = 25
    fmax_low = 8
    
    discrete_power_coefs = []
    
    # Compute Welch's coefficients for all 19 channels
    for x in range(len(EEG_closed)):
        f, Pxx = WelchEstimate(EEG_closed[x], c.fs, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax_wide)
        discrete_power_coefs.append(Pxx)

    # Pwide, Plow = [2,25], [2,8] Hz
    f_wide = f[(f >= fmin) & (f <= fmax_wide)]
    f_low = f[(f >= fmin) & (f <= fmax_low)]

    # Define slice indices to bound the frequency spectrum
    start_wide = f.tolist().index(f_wide[0])        
    start_low = f.tolist().index(f_low[0])   
    end_wide = f.tolist().index(f_wide[-1])+1 
    end_low = f.tolist().index(f_low[-1])+1 

    # Sum the total power for Pwide and Plow
    Pwide = np.array(discrete_power_coefs)[:,start_wide:end_wide].sum(axis=0).sum()
    Plow = np.array(discrete_power_coefs)[:,start_low:end_low].sum(axis=0).sum()

    # Compute the power ratio between Pwide and Plow
    Qslow = Plow / Pwide

    return float(Qslow)


def QASYM(EEG, channels):
    
    '''Computes asymmetrical background patterns by comparing rhythmic activity between the two hemispheres in corresponding
       left and right channel pairs. For each lr-channel pair (8x) a normalised value is computed. 
       The final asymmetry value is found by normalising the individual asymmetry values for each channel pair.
    
    Parameters
    ----------
    EEG : numpy array
         Multidimensional array, where (rows = channels, columns = timepoints).
         Input array requires a re-referenced laplacian montage.
         
    channels : numpy array
         1D array with the ordered channel names.

    Returns
    -------
    Qasym : float
         The final output is a quantified and normalised value for asymmetry for left and right hemispheres.
         
         If Qasym > 0.5: Normal asymmetry.
         If Qasym < 0.5: Abnormal asymmetry.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007

    '''
    
    assert len(EEG) and len(channels) == 19, "Exception: Expected 19 channels"
    assert len(EEG) == len(channels), "Exception: Returned Shape Mismatch"
    assert type(EEG) and type(channels) == np.ndarray, "Exception: Returned Type Mismatch"
    assert EEG.shape[1] >= 250*5, "Exception: To few data-points recorded"

    fmin = 0.5
    fmax = 15
    
    left  = ['fp1','f7','f3','t3','c3','t5','p3','o1']
    right = ['fp2','f8','f4','t4','c4','t6','p4','o2']

    discrete_power_coefs_left = []
    discrete_power_coefs_right = []
    
    # Compute Welch's coefficients for all left-right channels
    for x in range(channels.shape[0]):
        if channels[x].lower() in left:
            _, Pxx = WelchEstimate(EEG[x], c.fs, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
            discrete_power_coefs_left.append(Pxx)
        if channels[x].lower() in right:
            _, Pxx = WelchEstimate(EEG[x], c.fs, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
            discrete_power_coefs_right.append(Pxx)

    lr_asymmetries = []
    
    # Compute asymmetries between left and right pairs
    for pair in range(len(discrete_power_coefs_right)):
        L = np.array(discrete_power_coefs_left)[pair].sum()
        R = np.array(discrete_power_coefs_right)[pair].sum()
        lr_asymmetries.append( (R-L)/(R+L) )
        
    # Compute normalised asymmetry value
    Qasym = np.sum(np.abs(lr_asymmetries))/len(lr_asymmetries)
        
    return float(Qasym)


def QREAC(EEG_closed, EEG_open):
    
    '''Computes reactivity of the Alpha band [8-12]Hz.
       Using the estimated PDR peak value, the reactivity is calculated by
       constructing a 0.5 Hz frequency band on the estimated dominant peak frequency
       when the eyes are open and when the eyes are closed. Based on these values
       a normalised value is found which quantifies the reactivity of the PDR.
     
    Parameters
    ----------
    EEG_closed : numpy array
         Multidimensional array, where (rows = channels, columns = timepoints).
         Input array requires a common reference montage, using eyes closed states only.
    
    EEG_open: numpy array
         Multidimensional array, where (rows = channels, columns = timepoints).
         Input array requires a common reference montage, using eyes open states only.

    Returns
    -------
    Qreac : float
         The final output is a quantified and normalised value for alpha power reactivity.
        
        If Qreac > 0.5: Substantial reactivity. 
        If 0.1 < Qreac < 0.5: low reactivity.
        If Qreac < 0.1: absent reactivity.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007
    
    '''
    
    assert len(EEG_closed) and len(EEG_open) == 19, "Exception: Expected 19 channels"
    assert len(EEG_closed) == len(EEG_open), "Exception: Returned Shape Mismatch"
    assert type(EEG_closed) and type(EEG_open) == np.ndarray, "Exception: Returned Type Mismatch"
    assert EEG_closed.shape[1] and EEG_open.shape[1] >= 250*5, "Exception: To few datapoints recorded"

    fmin = 8
    fmax = 12

    o1 = 18
    o2 = 4
    
    # Compute Welch's coefficients for all eyes closed and open segments, for both channels O1 and O2
    _, Pxx_closed_o1 = WelchEstimate(EEG_closed[o1], c.fs, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
    _, Pxx_closed_o2 = WelchEstimate(EEG_closed[o2], c.fs, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
    _, Pxx_open_o1 = WelchEstimate(EEG_open[o1], c.fs, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
    _, Pxx_open_o2 = WelchEstimate(EEG_open[o2], c.fs, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
    
    # Compute the mean power spectrum for eyes closed and eyes open states.
    Pxx_closed = np.array([Pxx_closed_o1, Pxx_closed_o2]).sum(axis=0) * 0.5
    Pxx_open = np.array([Pxx_open_o1, Pxx_open_o2]).sum(axis=0) * 0.5

    # Compute the mean power of the eyes open and eyes closed segments
    Pec = Pxx_closed.sum()
    Peo = Pxx_open.sum()
    
    # Compute the power ratio between eyes open and eyes closed state
    Qreac = 1 - (Peo/Pec)

    return float(Qreac)

def rsBSI(EEG, channels):
    '''Computes revised Brain Symmetry Index.
     
    Parameters
    ----------
    EEG : numpy array
         Multidimensional array, where (rows = channels, columns = timepoints).
         Input array requires a common reference montage.
         
    Returns
    -------
    rBSI : float
         The final output is a normalised value known as the revised Brain Symmetry Index.

    References
    ----------
    .. [1] Van Putten (2007), The revised brain symmetry index.
    .. DOI https://doi.org/10.1016/j.clinph.2007.07.019
    
    '''
    
    assert len(EEG) and len(channels) == 19, "Exception: Expected 19 channels"
    assert len(EEG) == len(channels), "Exception: Returned Shape Mismatch"
    assert type(EEG) and type(channels) == np.ndarray, "Exception: Returned Type Mismatch"
    assert EEG.shape[1] >= 250*5, "Exception: To few data-points recorded"

    fmin = 0.5
    fmax = 25
    
    left_outer  = ['o1','t5','t3','f7']
    left_mid    = ['o1','p3','c3','f3']
    right_mid   = ['o2','p4','c4','f4']
    right_outer = ['o2','t6','t4','f8']
    
    P_left_outer, P_left_mid  = [],[]
    P_right_outer, P_right_mid = [],[]

    # Compute Welch's coefficients for all left and right chains
    for x in range(EEG.shape[0]):
        if channels[x].lower() in left_outer:
            _, Pxx = WelchEstimate(EEG[x], c.fs, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
            P_left_outer.append(Pxx)
        if channels[x].lower() in left_mid:
            _, Pxx = WelchEstimate(EEG[x], c.fs, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
            P_left_mid.append(Pxx)
        if channels[x].lower() in right_mid:
            _, Pxx = WelchEstimate(EEG[x], c.fs, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
            P_right_mid.append(Pxx)
        if channels[x].lower() in right_outer:
            _, Pxx = WelchEstimate(EEG[x], c.fs, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
            P_right_outer.append(Pxx)
    
    # Compute the squared values of the estimated Fourier coefficients for left and right channels
    Ln = (np.square(P_left_outer).sum() + np.square(P_left_mid).sum()) *0.5
    Rn = (np.square(P_right_outer).sum() + np.square(P_right_mid).sum()) *0.5
    
    # Compute the ratio between the power for left and right channels
    rsbsi = (Rn-Ln)/(Rn+Ln)
    
    return float(rsbsi)
    
   
def spectrum_eeg(EEG):
    discrete_power_coefs = []
    for i in range(EEG.shape[0]):
        f, Pxx = WelchEstimate(EEG[i],fs=c.fs,
                               window='hann',
                               noverlap=c.NOVERLAP,
                               nfft=c.NFFT,
                               nperseg=c.WINDOW)
        discrete_power_coefs.append(Pxx)

    # output from source: X=[Fp2 F8 T4 T6 O2 F4 C4 P4 Fz Cz Pz F3 C3 P3 Fp1 F7 T3 T5 O1] 
    # print(name)
    X = np.array([0, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, -2, -2, -2, -1])  # x-direction
    Y = np.array([0, 1, 0, -1, -2, 1, 0, -1, 1, 0, -1, 1, 0, -1, 0, 1, 0, -1, -2])  # y-direction
    P = np.array(discrete_power_coefs).T
    # print(P.shape,X.shape,Y.shape)

    XP=(P @ X)/P.sum(axis=1)
    YP=(P @ Y)/P.sum(axis=1)

    COG = np.array([XP,YP]).T
    
    return P, COG, f


def signal_coherence(EEG, fs, WINDOW, NOVERLAP, NFFT):

    fp2, f8, t4, t6, o2, f4, c4, p4, fz, cz = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    pz, f3, c3, p3, fp1, f7, t3, t5, o1 = 10, 11, 12, 13, 14, 15, 16, 17, 18

    f, FP1a = coherence(EEG[fp1], EEG[fz], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, FP1b = coherence(EEG[fp1], EEG[f3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, FP1c = coherence(EEG[fp1], EEG[f7], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cFp1 = np.mean([FP1a, FP1b, FP1c])

    _, FP2a = coherence(EEG[fp2], EEG[fz], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, FP2b = coherence(EEG[fp2], EEG[f4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, FP2c = coherence(EEG[fp2], EEG[f8], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cFp2 = np.mean([FP2a, FP2b, FP2c])

    _, F4a = coherence(EEG[f4], EEG[fz], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, F4b = coherence(EEG[f4], EEG[f8], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, F4c = coherence(EEG[f4], EEG[c4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cF4 = np.mean([F4a, F4b, F4c])

    _, F3a = coherence(EEG[f3], EEG[fz], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, F3b = coherence(EEG[f3], EEG[f7], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, F3c = coherence(EEG[f3], EEG[c3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cF3 = np.mean([F3a, F3b, F3c])

    _, Fza = coherence(EEG[fz], EEG[f3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, Fzb = coherence(EEG[fz], EEG[f4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, Fzc = coherence(EEG[fz], EEG[cz], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cFz = np.mean([Fza, Fzb, Fzc])

    _, Cza = coherence(EEG[cz], EEG[fz], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, Czb = coherence(EEG[cz], EEG[c4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, Czc = coherence(EEG[cz], EEG[c3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, Czd = coherence(EEG[cz], EEG[pz], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cCz = np.mean([Cza, Czb, Czc, Czd])

    _, Pza = coherence(EEG[pz], EEG[p4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, Pzb = coherence(EEG[pz], EEG[p3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, Pzc = coherence(EEG[pz], EEG[o2], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, Pzd = coherence(EEG[pz], EEG[o1], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cPz = np.mean([Pza, Pzb, Pzc, Pzd])

    _, C4a = coherence(EEG[c4], EEG[cz], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, C4b = coherence(EEG[c4], EEG[f4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, C4c = coherence(EEG[c4], EEG[p4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, C4d = coherence(EEG[c4], EEG[t4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cC4 = np.mean([C4a, C4b, C4c, C4d])

    _, C3a = coherence(EEG[c3], EEG[cz], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, C3b = coherence(EEG[c3], EEG[f3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, C3c = coherence(EEG[c3], EEG[p3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, C3d = coherence(EEG[c3], EEG[t3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cC3 = np.mean([C3a, C3b, C3c, C3d])

    _, P4a = coherence(EEG[p4], EEG[c4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, P4b = coherence(EEG[p4], EEG[t6], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, P4c = coherence(EEG[p4], EEG[pz], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, P4d = coherence(EEG[p4], EEG[o2], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cP4 = np.mean([P4a, P4b, P4c, P4d])

    _, P3a = coherence(EEG[p3], EEG[c3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, P3b = coherence(EEG[p3], EEG[t5], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, P3c = coherence(EEG[p3], EEG[pz], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, P3d = coherence(EEG[p3], EEG[o1], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cP3 = np.mean([P3a, P3b, P3c, P3d])

    _, O2a = coherence(EEG[o2], EEG[pz], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, O2b = coherence(EEG[o2], EEG[p4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, O2c = coherence(EEG[o2], EEG[t6], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cO2 = np.mean([O2a, O2b, O2c])

    _, O1a = coherence(EEG[o1], EEG[pz], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, O1b = coherence(EEG[o1], EEG[p3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, O1c = coherence(EEG[o1], EEG[t5], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cO1 = np.mean([O1a, O1b, O1c])

    _, F8a = coherence(EEG[f8], EEG[t4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, F8b = coherence(EEG[f8], EEG[c4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, F8c = coherence(EEG[f8], EEG[f4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cF8 = np.mean([F8a, F8b, F8c])

    _, F7a = coherence(EEG[f7], EEG[t3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, F7b = coherence(EEG[f7], EEG[c3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, F7c = coherence(EEG[f7], EEG[f3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cF7 = np.mean([F7a, F7b, F7c])

    _, T4a = coherence(EEG[t4], EEG[f8], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, T4b = coherence(EEG[t4], EEG[c4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, T4c = coherence(EEG[t4], EEG[t6], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cT4 = np.mean([T4a, T4b, T4c])

    _, T3a = coherence(EEG[t3], EEG[f7], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, T3b = coherence(EEG[t3], EEG[c3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, T3c = coherence(EEG[t3], EEG[t5], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cT3 = np.mean([T3a, T3b, T3c])

    _, T6a = coherence(EEG[t6], EEG[t4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, T6b = coherence(EEG[t6], EEG[p4], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, T6c = coherence(EEG[t6], EEG[o2], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cT6 = np.mean([T6a, T6b, T6c])

    _, T5a = coherence(EEG[t5], EEG[t3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, T5b = coherence(EEG[t5], EEG[p3], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    _, T5c = coherence(EEG[t5], EEG[o1], fs, 'hann', WINDOW, NOVERLAP, NFFT)
    cT5 = np.mean([T5a, T5b, T5c])

    # maximum coherence value from all 19 NN coherence's at each frequency
    AllCOH = [cFp2, cF8, cT4, cT6, cO2, cF4, cC4, cP4, cFz, cCz, cPz, cF3, cC3, cP3, cFp1, cF7, cT3, cT5, cO1]
    max_coh_value = np.max(AllCOH)

    return AllCOH, max_coh_value, f
