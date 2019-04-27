import pycwt
from python_FC_pkg import fc_utils, clustering
import numpy as np


def mean_resultant_length(angle_vector):
    #functions are taken from the circular data analysis chapter
    C=np.sum(np.cos(angle_vector))
    S=np.sum(np.sin(angle_vector))
    return np.sqrt(C**2+S**2)/len(angle_vector)

def circular_standard_deviation(angle_vector):
    #functions are taken from the circular data analysis chapter
    return np.sqrt(-2*np.log(mean_resultant_length(angle_vector)))

def cross_wavelet_transform(y1, y2, dt, dj=1/12, s0=-1, J=-1,
                            wavelet='morlet', normalize=True):

    wavelet = pycwt.wavelet._check_parameter_wavelet(wavelet)

    # Makes sure input signal are numpy arrays.
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    # Calculates the standard deviation of both input signals.
    std1 = y1.std()
    std2 = y2.std()
    # Normalizes both signals, if appropriate.
    if normalize:
        y1_normal = (y1 - y1.mean()) / std1
        y2_normal = (y2 - y2.mean()) / std2
    else:
        y1_normal = y1
        y2_normal = y2

    # Calculates the CWT of the time-series making sure the same parameters
    # are used in both calculations.
    _kwargs = dict(dj=dj, s0=s0, J=J, wavelet=wavelet)
    W1, sj, freq, coi, _, _ = pycwt.cwt(y1_normal, dt, **_kwargs)
    W2, sj, freq, coi, _, _ = pycwt.cwt(y2_normal, dt, **_kwargs)

    # Calculates the cross CWT of y1 and y2.
    W12 = W1 * W2.conj()

    return W12, coi, freq

def restrict_time_to_coi(xWT, coi, min_freq):
    #this function will cut out timepoints which are outside the coi of the minimal frequency specified
    freqs_coi=1/coi
    indices=min_freq>freqs_coi #simply determine for which timepoints the min_freq is within the coi
    return xWT[:,indices]

def full_time_freq_coupling(timeseries,num_nodes,dt,high_f=0.11,low_f=0.005, xWT_freq_threshold=0.005, coi_freq_threshold=0.005, num_sub_octaves=12, verbose=True):
    mother = pycwt.Morlet(6) #setting a Morlet wavelet with omega=6, which sets the time-frequency resolution of the wavelet and is recommended in Chang and Glover
    s0 = 1/high_f
    dj = 1 / num_sub_octaves
    num_powers=0
    while((1/high_f)*(2**num_powers)<1/low_f):
        num_powers+=1
    J = int(num_powers / dj)

    y1=timeseries[:,0]
    y2=timeseries[:,1]
    xWT, coi, freq=cross_wavelet_transform(y1, y2, dt, dj=dj, s0=s0, J=J, wavelet=mother, normalize=True)
    xWT=xWT[freq>=xWT_freq_threshold,:]
    coi_xWT=restrict_time_to_coi(xWT,coi,coi_freq_threshold) # restrict timepoints to only where min freq is met
    full_coupling=np.zeros([int((num_nodes**2-num_nodes)/2),coi_xWT.shape[0],coi_xWT.shape[1]], dtype='complex128')

    i=0
    for roi1 in range(num_nodes):
        for roi2 in range(roi1):
            y1=timeseries[:,roi1]
            y2=timeseries[:,roi2]
            xWT, coi, freq=cross_wavelet_transform(y1, y2, dt, dj=dj, s0=s0, J=J, wavelet=mother, normalize=True)
            xWT=xWT[freq>=xWT_freq_threshold,:]
            coi_xWT=restrict_time_to_coi(xWT,coi,coi_freq_threshold) # restrict timepoints to only where min freq is met
            full_coupling[i,:,:]=coi_xWT
            if verbose:
                print('Edge # '+str(i))
            i+=1
    return full_coupling





def time_averaged_xWT(y1,y2,dt,avg_type='phased',high_f=0.11,low_f=0.005,min_scan_time=6*60):
    if avg_type=='phased':
        import pycwt
        mother = pycwt.Morlet(6) #setting a Morlet wavelet with omega=6, which sets the time-frequency resolution of the wavelet and is recommended in Chang and Glover
        s0 = 1/high_f
        dj = 1 / 12  # Twelve sub-octaves per octaves
        num_powers=0
        while((1/high_f)*(2**num_powers)<1/low_f):
            num_powers+=1
        J = int(num_powers / dj)

        xWT, coi, freq=cross_wavelet_transform(y1, y2, dt, dj=dj, s0=s0, J=J, wavelet=mother, normalize=True)
        xWT_amp=np.abs(xWT)
        xWT_ang=np.angle(xWT)
        xWT_phased=xWT_amp*np.cos(xWT_ang)

        freqs_coi=1/coi
        #threshold the frequencies at the max frequencies of the wavelet
        pos_indices=freqs_coi>np.max(freq)
        neg_indices=freqs_coi<np.min(freq)
        for i in range(len(freqs_coi)):
            if pos_indices[i]:
                freqs_coi[i]=np.max(freq)
            elif neg_indices[i]:
                freqs_coi[i]=np.min(freq)

        within_freqs_coi=np.zeros(xWT.shape)
        for i in range(len(freqs_coi)):
            freq_index=0
            while(freq[freq_index]>freqs_coi[i]):
                freq_index+=1
            within_freqs_coi[:freq_index,i]=1

        num_coi_points=np.sum(within_freqs_coi,1)
        valid_freq=num_coi_points>min_scan_time/dt
        if not valid_freq[0]:
            raise ValueError('The minimum number of timepoints within the cone of influence is not met for any frequencies.')

        xWT_phased_avg=np.zeros(np.sum(valid_freq))
        xWT_phased_var=np.zeros(np.sum(valid_freq))
        xWT_amp_avg=np.zeros(np.sum(valid_freq))
        xWT_amp_var=np.zeros(np.sum(valid_freq))
        xWT_ang_avg=np.zeros(np.sum(valid_freq))
        xWT_ang_var=np.zeros(np.sum(valid_freq))
        #calculate the temporal means and variabilities within the coi at frequencies
        # which respect the minimum time length requirement
        for i in range(len(valid_freq)):
            if valid_freq[i]:
                #get the indices of data falling within the coi
                indices=[j for j, x in enumerate(within_freqs_coi[i,:]) if x]
                xWT_phased_avg[i]=np.mean(xWT_phased[i,indices])
                xWT_phased_var[i]=np.std(xWT_phased[i,indices])
                xWT_amp_avg[i]=np.mean(xWT_amp[i,indices])
                xWT_amp_var[i]=np.std(xWT_amp[i,indices])
                xWT_ang_avg[i]=mean_resultant_length(xWT_ang[i,indices])
                xWT_ang_var[i]=circular_standard_deviation(xWT_ang[i,indices])

        return xWT_phased_avg, xWT_phased_var, xWT_amp_avg,xWT_amp_var,xWT_ang_avg,xWT_ang_var,freq[:np.sum(valid_freq)]

#    elif avg_type=='quadrants':
        #this type will calculate a different average for each quadrants of the phase in the WCT
    return None

def full_time_averaged_xWT(timeseries,dt,avg_type='phased',high_f=0.11,low_f=0.005,min_scan_time=6*60):
    #evaluate both the temporal variability across both the xWT amplitude and angle
    xWT_amplitude_variability=[]
    xWT_phase_variability=[]
    xWT_amplitude_avg=[]
    xWT_phase_avg=[]
    xWT_phasedAmp_variability=[]
    xWT_phasedAmp_avg=[]

    for roi1 in range(num_nodes):
        for roi2 in range(roi1):
            y1=sub_timeseries[:,roi1]
            y2=sub_timeseries[:,roi2]
            xWT_phased_avg, xWT_phased_var, xWT_amp_avg,xWT_amp_var,xWT_ang_avg,xWT_ang_var,valid_freqs=time_averaged_xWT(y1,y2,dt,avg_type='phased',high_f=high_f,low_f=low_f,min_scan_time=min_scan_time)
            xWT_phasedAmp_avg.append(xWT_phased_avg)
            xWT_phasedAmp_variability.append(xWT_phased_var)
            xWT_amplitude_avg.append(xWT_amp_avg)
            xWT_amplitude_variability.append(xWT_amp_var)
            xWT_phase_avg.append(xWT_ang_avg)
            xWT_phase_variability.append(xWT_ang_var)

    return np.asarray(xWT_phasedAmp_avg,dtype='float64'), np.asarray(xWT_phasedAmp_variability,dtype='float64'), np.asarray(xWT_amplitude_avg,dtype='float64'), np.asarray(xWT_amplitude_variability,dtype='float64'), np.asarray(xWT_phase_avg,dtype='float64'), np.asarray(xWT_phase_variability,dtype='float64'),valid_freqs
