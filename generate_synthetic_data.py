import random
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


def random_delta(size, start_value, min_delta, max_delta, precision=1):
    value_range = max_delta - min_delta
    half_range = value_range / 2
    for _ in range(size):
        delta = random.random() * value_range - half_range
        yield round(start_value, precision)
        start_value += delta

def scale_array(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    # Scaling between 0 and 1, then to [-1, 1]
    scaled_arr = 2 * (arr - arr_min) / (arr_max - arr_min) - 1
    return scaled_arr

def generate_clean_signal(length, min_delta=-0.1, max_delta=0.1, precision=100):
    """
    Generate a clean signal with random fluctuations.

    Args:
    length (int): The length of the signal.
    min_delta (float): Minimum change in value.
    max_delta (float): Maximum change in value.
    precision (int): Precision of the values.

    Returns:
    numpy.ndarray: Clean signal.
    """
    signal = scale_array(savgol_filter(list(random_delta(length, 0, min_delta, max_delta, precision)), 20, 3))
    return np.array(signal)


def generate_noise_profile(length, num_points=10, min_level=0.05, max_level=0.2):
    """
    Generate a noise profile with varying levels.
    
    Args:
    length (int): The length of the signal.
    num_points (int): Number of points to use for interpolation.
    min_level (float): Minimum noise level.
    max_level (float): Maximum noise level.
    
    Returns:
    numpy.ndarray: Noise level profile for the entire signal length.
    """
    # Generate random noise levels at specific points
    x = np.linspace(0, length - 1, num_points)
    y = np.random.uniform(min_level, max_level, num_points)
    
    # Interpolate to get noise levels for all points
    f = interp1d(x, y, kind='cubic')
    x_new = np.arange(length)
    noise_profile = f(x_new)
    
    return noise_profile

def add_noise(signal, base_noise_level=0.1, variable_noise_factor=0.5):
    """
    Add random noise with varying levels to the signal.
    
    Args:
    signal (numpy.ndarray): The input signal.
    base_noise_level (float): The base level of noise to add.
    variable_noise_factor (float): Factor to determine the variability of noise (0 to 1).
    
    Returns:
    numpy.ndarray: Noisy signal.
    """
    length = len(signal)
    
    # Generate noise profile
    noise_profile = generate_noise_profile(length, 
                                           min_level=base_noise_level * (1 - variable_noise_factor),
                                           max_level=base_noise_level * (1 + variable_noise_factor))
    
    # Generate noise and scale it according to the profile
    noise = np.random.normal(0, 1, length) * noise_profile
    
    noisy_signal = signal + noise
    return noisy_signal

def scale_to_range(clean_signal, noisy_signal, min_val=-1, max_val=1):
    """
    Scale the clean and noisy signals to a specific range.

    Args:
    clean_signal (numpy.ndarray): Clean signal.
    noisy_signal (numpy.ndarray): Noisy signal.
    min_val (float): Minimum value for scaling.
    max_val (float): Maximum value for scaling.

    Returns:
    numpy.ndarray: Scaled clean signal.
    numpy.ndarray: Scaled noisy signal.
    """
    noisy_signal_min = np.min(noisy_signal)
    noisy_signal_max = np.max(noisy_signal)
    clean_signal_scaled = (max_val - min_val) * (clean_signal - noisy_signal_min) / (noisy_signal_max - noisy_signal_min) + min_val
    noisy_signal_scaled = (max_val - min_val) * (noisy_signal - noisy_signal_min) / (noisy_signal_max - noisy_signal_min) + min_val
    return clean_signal_scaled, noisy_signal_scaled

def generate_dataset(num_samples, sequence_length):
    """
    Generate a dataset of clean and noisy signals.

    Args:
    num_samples (int): Number of samples to generate.
    sequence_length (int): Length of each signal.

    Returns:
    numpy.ndarray: Clean signals.
    numpy.ndarray: Noisy signals.
    """
    clean_signals = []
    noisy_signals = []
    
    for _ in range(num_samples):
        clean_signal = generate_clean_signal(sequence_length, -0.1, 0.1)
        noisy_signal = add_noise(clean_signal, np.random.uniform(0.1, 0.5), np.random.uniform(0.5, 1.0))

        clean_signal_scaled, noisy_signal_scaled = scale_to_range(clean_signal, noisy_signal)

        clean_signals.append(clean_signal_scaled)
        noisy_signals.append(noisy_signal_scaled)
    
    return np.array(clean_signals), np.array(noisy_signals)
