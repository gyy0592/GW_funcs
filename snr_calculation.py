import matplotlib.pyplot as plt
import numpy as np 
from scipy import signal
from scipy.interpolate import interp1d


# Function to normalize the SNR of a signal to a target SNR within a specified range
def normalize_snr(signal, snr_target,psd):
    # Calculate the current SNR of the signal within the specified range
    signal_ft = np.fft.fft(signal) / np.sqrt(len(signal)) 
    snr_current = calculate_snr(signal_ft,signal_ft,psd)
    # Calculate the scaling factor required to achieve the target SNR
    scaling_factor = (snr_target / snr_current)
    # Normalize the signal by multiplying it with the scaling factor
    normalized_signal = signal * scaling_factor
    return normalized_signal


def compute_inner_product(a, b, psd, df=1./4096, freq_range=None, domain='fourier'):
    N = len(a)  # Assuming a and b have the same length

    # Convert to Fourier domain if inputs are in time domain
    if domain == 'time':
        a = np.fft.fft(a)/np.sqrt(len(a))
        b = np.fft.fft(b)/np.sqrt(len(b))

    # Frequency array for a and b, considering the symmetry of FFT
    freqs_data = np.fft.fftfreq(len(a,),df)

    # Frequency array for PSD (only positive frequencies, as it's real-valued)
    freqs_psd = np.linspace(0, 1/(2*df), len(psd))

    conj_a = np.conjugate(a)

    # Interpolate psd to match the length of a and b
    psd_interpolation_func = interp1d(freqs_psd, psd, kind='linear', fill_value=np.nan)

    # Limit the computation to the specified frequency range, if provided
    if freq_range is not None:
        freq_data_indices = np.where((np.abs(freqs_data) > freq_range[0]) & (np.abs(freqs_data) < freq_range[1]))[0]
        freqs_psd_indices = np.where((np.abs(freqs_psd) > freq_range[0]) & (np.abs(freqs_psd) < freq_range[1]))[0]
        conj_a = conj_a[freq_data_indices]
        b = b[freq_data_indices]
        # freqs_data = freqs_data[freq_data_indices]
    else:
        freq_data_indices = np.arange(len(freqs_data))

    psd_interpolated = psd_interpolation_func(np.abs(freqs_data[freq_data_indices]))
    psd_interpolated = np.clip(psd_interpolated, np.min(psd[freqs_psd_indices]), np.max(psd[freqs_psd_indices]))
    if np.isnan(psd_interpolated).any():
        print('Warning: NaN values in interpolated PSD')    
    integrand = conj_a * b / psd_interpolated*df
    return np.real(np.sum(integrand))

import cupy as cp

def compute_inner_product_gpu(a, b, psd, df=1./4096, freq_range=None, domain='fourier'):
    a = cp.asarray(a)
    b = cp.asarray(b)
    # Convert to Fourier domain if inputs are in time domain
    if domain == 'time':
        a = cp.fft.fft(a)/cp.sqrt(len(a))
        b = cp.fft.fft(b)/cp.sqrt(len(b))
    # convert back the numpy array
    a = cp.asnumpy(a)
    b = cp.asnumpy(b)   
    # Frequency array for a and b, considering the symmetry of FFT
    freqs_data = np.fft.fftfreq(len(a,),df)

    # Frequency array for PSD (only positive frequencies, as it's real-valued)
    freqs_psd = np.linspace(0, 1/(2*df), len(psd))

    conj_a = np.conjugate(a)

    # Interpolate psd to match the length of a and b
    # psd_interpolation_func = cp_interp1d(freqs_psd, psd, kind='linear', fill_value=np.nan)

    # Limit the computation to the specified frequency range, if provided
    if freq_range is not None:
        freq_data_indices = np.where((np.abs(freqs_data) > freq_range[0]) & (np.abs(freqs_data) < freq_range[1]))[0]
        conj_a = conj_a[freq_data_indices]
        b = b[freq_data_indices]
        # freqs_data = freqs_data[freq_data_indices]

    psd_interpolated = cp.interp(cp.array(np.abs(freqs_data[freq_data_indices])),cp.array(freqs_psd), cp.array(psd))
    psd_interpolated = cp.asnumpy(psd_interpolated)
    # convert everything into cupy array with batching, batch size is 1e7 
    batch_size = int(1e8)
    num_batches = len(conj_a) // batch_size
    if len(conj_a) % batch_size != 0:
        num_batches += 1
    inner_product = 0
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        gpu_conj_a = cp.asarray(conj_a[start:end])
        gpu_b = cp.asarray(b[start:end])
        gpu_psd_interpolated = cp.asarray(psd_interpolated[start:end])
        inner_product += cp.sum(cp.real(gpu_conj_a * gpu_b / gpu_psd_interpolated*df))
    return cp.asnumpy(inner_product)



def compute_inner_product_gpu_gpuArray(a, b, psd, df=1./4096, freq_range=None, domain='fourier'):
    # Convert to Fourier domain if inputs are in time domain
    if domain == 'time':
        a = cp.fft.fft(a)/cp.sqrt(len(a))
        b = cp.fft.fft(b)/cp.sqrt(len(b))
    # convert back the numpy array
    a = cp.asnumpy(a)
    b = cp.asnumpy(b)   
    # Frequency array for a and b, considering the symmetry of FFT
    freqs_data = np.fft.fftfreq(len(a,),df)

    # Frequency array for PSD (only positive frequencies, as it's real-valued)
    freqs_psd = cp.linspace(0, 1/(2*df), len(psd))

    conj_a = np.conjugate(a)

    # Interpolate psd to match the length of a and b
    # psd_interpolation_func = cp_interp1d(freqs_psd, psd, kind='linear', fill_value=np.nan)

    # Limit the computation to the specified frequency range, if provided
    if freq_range is not None:
        freq_data_indices = np.where((np.abs(freqs_data) > freq_range[0]) & (np.abs(freqs_data) < freq_range[1]))[0]
        conj_a = conj_a[freq_data_indices]
        b = b[freq_data_indices]
        # freqs_data = freqs_data[freq_data_indices]

    if len(freqs_data[freq_data_indices]) == len(freqs_psd):
        psd_interpolated = psd
    else:
        psd_interpolated = cp.interp(cp.array(np.abs(freqs_data[freq_data_indices])), cp.array(freqs_psd), psd)
    # convert everything into cupy array with batching, batch size is 1e7 
    batch_size = int(1e8)
    num_batches = len(conj_a) // batch_size
    if len(conj_a) % batch_size != 0:
        num_batches += 1
    inner_product = 0
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        gpu_conj_a = cp.asarray(conj_a[start:end])
        gpu_b = cp.asarray(b[start:end])
        gpu_psd_interpolated = cp.asarray(psd_interpolated[start:end])
        inner_product += cp.sum(cp.real(gpu_conj_a * gpu_b / gpu_psd_interpolated * df))
    return cp.asnumpy(inner_product)

def calculate_snr_gpu(signal_ft, template_ft, noise_psd, df=1./4096,freq_range=[30,1700],domain='fourier'):
    # return numpy results
	return np.sqrt(np.abs(compute_inner_product_gpu(signal_ft, template_ft, noise_psd, df, freq_range, domain)))
	

def calculate_snr(signal_ft, template_ft, noise_psd, df=1./4096,freq_range=[30,1700],domain='fourier'):
	return np.sqrt(np.abs(compute_inner_product(signal_ft, template_ft, noise_psd, df, freq_range, domain)))
	
def calculate_snr_gpuArray(signal_ft_gpuArry, template_ft_gpuArray, noise_psd_gpuArray, df=1./4096,freq_range=[30,1700],domain='fourier'):
    # return numpy results
	return np.sqrt(np.abs(compute_inner_product_gpu_gpuArray(signal_ft_gpuArry, template_ft_gpuArray, noise_psd_gpuArray, df, freq_range, domain)))
	

def matched_filter(signal, template, noise_asd, freq_range=None, domain='time', step=1):
    # Length of the signal
    signal_length = len(signal)
    
    # Pre-allocate the result array for the matched filter output
    result_length = (signal_length + step - 1) // step
    result = np.zeros(result_length)
    
    # Perform the convolution/matched filtering with a step size
    for i in range(0, signal_length, step):
        # Shift template to align with the current position in the signal
        shifted_template = np.roll(template, i)
        
        # Compute the inner product using the provided function
        result[i // step] = compute_inner_product(signal, shifted_template, noise_asd**2, freq_range=freq_range, domain = domain)
    
    return result