import numpy as np 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

def generate_noise_from_psd(N, psd, num_noises=1,freq_range=[30,1700], sample_rate = 4096, duration = None):
    freqVec = np.linspace(0, sample_rate/2, len(psd))
    noises = np.zeros((num_noises, N))
    random_asds = []
    for i in range(num_noises):
        WGN = np.random.randn(N)
        X = np.fft.rfft(WGN) / np.sqrt(N)
        asd = np.sqrt(psd)
        uneven = N % 2
        # Simulate the white noise of rFFT
        # X = (np.random.randn(N // 2 + 1 + uneven) + 1j * np.random.randn(N // 2 + 1 + uneven))
        
        selected_indices = np.where((freqVec > freq_range[0]) & (freqVec < freq_range[1]))[0]
        
        # Interpolate selected ASD values to match the length of X
        interp_asd = interp1d(freqVec[selected_indices],asd[selected_indices], kind='linear', bounds_error=False, fill_value="extrapolate")
        newFreqVec = np.fft.rfftfreq(N+uneven, d=1.0/sample_rate)
        random_asd = interp_asd(newFreqVec)
        nonSelected_indices = np.where(~ ((newFreqVec> freq_range[0]) & (newFreqVec < freq_range[1])))[0]
        random_asd[nonSelected_indices] = 1e-30
        # random_asd[random_asd<1e-30] = 1e-30

        # Apply the random ASD to create colored noise
        # In order to keep the nSample equal to before
        Y_colored = X * random_asd
        y_colored = np.fft.irfft(Y_colored).real * np.sqrt(N*sample_rate)
        if uneven:
            y_colored = y_colored[:-1]
        
        noises[i, :] = y_colored 
        random_asds.append(random_asd)
        
    return noises, random_asds

def generate_noise_from_psd_v2(psd, num_noises=1, freq_range=[30, 1700], sample_rate=4096):
    N = int(len(psd))
    freqVec = np.linspace(0, sample_rate/2, N)
    noises = np.zeros((num_noises, N))
    random_asds = []
    
    for i in range(num_noises):
        asd = np.sqrt(psd)
        uneven = N % 2
        # X = np.random.randn(N // 2 + 1 + uneven) + 1j * np.random.randn(N // 2 + 1 + uneven)
        X = np.random.randn(N) + 1j * np.random.randn(N)

        random_asd = asd.copy()
        # random_asd[~selected_indices] = 1e-30
        
        factor = np.mean(np.abs(X))
        Y_colored = X * random_asd[:len(X)] / factor
        y_colored = np.fft.irfft(Y_colored).real * np.sqrt(len(Y_colored) * sample_rate)
        if uneven:
            y_colored = y_colored[:-1]
        
        noises[i, :] = y_colored[:N] 
        random_asds.append(random_asd)
    
    return noises, random_asds

def generate_noise_from_psd_v3(psd, num_noises=1, freq_range=[30, 1700], duration = 10, sample_rate=4096):
    N = int(len(psd))
    full_N = 2 * (N - 1)  # Length of the time series should be twice the length of the input minus 2 plus 1 (Nyquist)
    freqVec = np.linspace(0, sample_rate / 2, N)
    
    selected_nSamples = int(duration * sample_rate)
    noises = np.zeros((num_noises, selected_nSamples))
    random_asds = []

    for i in range(num_noises):
        WGN = np.random.randn(full_N)
        X = np.fft.rfft(WGN)/np.sqrt(len(WGN))
        asd = np.sqrt(psd)
        
        X[0] = np.random.randn()  # Real part for DC component
        if N % 2 == 0:
            X[-1] = np.random.randn()  # Real part for Nyquist component
        
        random_asd = asd.copy()
        
        # Apply the ASD to the random complex numbers
        Y_colored = X * random_asd
        
        # Construct the full spectrum (including negative frequencies)
        Y_colored_full = np.concatenate([Y_colored, np.conj(Y_colored[-2:0:-1])])
        # Inverse FFT to get the time series data
        y_colored = np.fft.ifft(Y_colored_full, n=full_N).real * np.sqrt(full_N * sample_rate) 
        
        # select middle 50 seconds of y_colored
        # noises[i, :] = y_colored

        start_idx = full_N//2-selected_nSamples//2
        end_idx = start_idx + selected_nSamples
        noises[i, :] = y_colored[start_idx:end_idx]
        random_asds.append(random_asd)
        
    return noises, random_asds



def generate_asd_list(data_list,fftlength = None, fs = 4096):
    # random_data_list = random.sample(data_list, 50)
    asd_list = []
    # for data in random_data_list:
    for data in data_list:
        ts = TimeSeries(data, sample_rate=fs)
        if fftlength is None:
            asd = ts.asd()
        else:
            asd = ts.asd(fftlength=fftlength)  # You might adjust fftlength based on your specific needs
        asd_list.append(asd)
    return asd_list



from scipy.interpolate import interp1d
from .LIGO_data import RunningMedian #, RunningMedian_gpu

def obtain_smooth_asd(asd, window_size):
    """
    Process the ASD by applying a running median and interpolating it.

    Parameters:
        asd (numpy.ndarray): Input ASD array.
        window_size (int): Window size for the running median.

    Returns:
        numpy.ndarray: Processed ASD array, same length as the original.
    """
    # Apply running median
    asd_smoothed = RunningMedian(asd, window_size)
    
    # Interpolate to match the original length
    x_original = np.arange(len(asd))
    x_smoothed = np.arange(len(asd_smoothed))
    interpolator = interp1d(x_smoothed, asd_smoothed, kind='linear', fill_value='extrapolate')
    asd_interpolated = interpolator(x_original)
    
    return asd_interpolated


def obtain_smooth_asd_gpu(asd, window_size):
    """
    Process the ASD by applying a running median and interpolating it using GPU acceleration.

    Parameters:
        asd (cupy.ndarray): Input ASD array.
        window_size (int): Window size for the running median.

    Returns:
        cupy.ndarray: Processed ASD array, same length as the original.
    """
    # Apply running median
    asd_smoothed = RunningMedian_gpu(asd, window_size)
    
    # Interpolate to match the original length
    x_original = cp.arange(len(asd))
    x_smoothed = cp.arange(len(asd_smoothed))
    interpolator = interp1d(cp.asnumpy(x_smoothed), cp.asnumpy(asd_smoothed), kind='linear', fill_value='extrapolate')
    asd_interpolated = cp.asarray(interpolator(cp.asnumpy(x_original)))
    
    return asd_interpolated


def generate_perturbed_psd(polynomial_fitting_function,optimized_coefficients,num_PSD):
    freqVec = np.linspace(0, 2048, 1000)
    inRangeIndices = (freqVec >= 30) & (freqVec <= 1700)
# Perturb the coefficients and generate perturbed PSDs
    perturbed_psds = []
    for _ in range(num_PSD):
        perturbation = np.random.uniform(0.98, 1.02, size=optimized_coefficients.shape)  # 10% perturbation
        perturbed_coefficients = optimized_coefficients * perturbation
        perturbed_psd = polynomial_fitting_function(freqVec, perturbed_coefficients)
        perturbed_psd[~inRangeIndices]=-999
        perturbed_psds.append(np.exp(perturbed_psd))
    return freqVec, perturbed_psds

def generate_perturbed_asd_noises(N, polynomial_fitting_function,optimized_coefficients,num_noises=1,sample_rate = 4096):
    """
    Generate multiple series of noise with ASD filtered by a threshold.
    Returns a (num_noises, N) array, each row containing a different noise series.
    """
    noises = np.empty((num_noises, N))
    random_asds = []
    freqVec, perturbed_psds = generate_perturbed_psd(polynomial_fitting_function,optimized_coefficients,num_noises)
    for i in range(num_noises):

        perturbed_psd = perturbed_psds[i]
        noise_list, noise_asd_list = generate_noise_from_psd(N, perturbed_psd, num_noises=1,freq_range=[30,1700], sample_rate = sample_rate)
        noises[i, :] = noise_list[0]
        # random_asds.append(np.sqrt(perturbed_psd))
        random_asds.append(noise_asd_list[0])
        
    return noises, random_asds

def polynomial_fitting_function(freq, params, num_polynomials=4, orders=5):
    x = np.log(freq / 300)
    # x = freq
    result = 0
    idx = 0
    for i in range(num_polynomials):
        for j in range(orders):
            result += params[idx] * x**j
            idx += 1
    return (result)