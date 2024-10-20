import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import random

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define the autoencoder architecture
def build_autoencoder(input_shape):
    # Encoder
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    encoded = layers.Dense(32, activation='relu')(x)

    # Decoder
    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(input_shape[0], activation='tanh')(x)

    # Create the autoencoder model
    autoencoder = models.Model(inputs, outputs)
    return autoencoder

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

def generate_realistic_signal(length, min_delta=-0.1, max_delta=0.1, precision=100):
    """Generate a realistic signal with random fluctuations."""
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

def generate_dataset(num_samples, sequence_length, base_noise_level=0.1, variable_noise_factor=0.5):
    """Generate a dataset of clean and noisy signals."""
    clean_signals = []
    noisy_signals = []
    
    for _ in range(num_samples):
        clean_signal = generate_realistic_signal(sequence_length, -0.1, 0.1)
        noisy_signal = add_noise(clean_signal, base_noise_level, variable_noise_factor)
        
        clean_signals.append(clean_signal)
        noisy_signals.append(noisy_signal)
    
    return np.array(clean_signals), np.array(noisy_signals)

num_samples = 1000
sequence_length = 1000
noise_level = 0.1
clean_signals, noisy_signals = generate_dataset(num_samples, sequence_length, noise_level)

# Build and compile the autoencoder
autoencoder = build_autoencoder((sequence_length,))
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(
    noisy_signals, clean_signals,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.2
)

# # Use the trained autoencoder to denoise some test data
# x_test = np.random.random((10, input_shape[0]))
# x_test_noisy = add_noise(x_test)
# denoised_images = autoencoder.predict(x_test_noisy)

# # Print some results
# print("Original shape:", x_test.shape)
# print("Noisy shape:", x_test_noisy.shape)
# print("Denoised shape:", denoised_images.shape)

# You can visualize the results using matplotlib if needed
# Function to plot results
def plot_results(clean_signal, noisy_signal, denoised_signal):
    plt.figure(figsize=(12, 4))
    plt.plot(clean_signal, label='Clean Signal')
    plt.plot(noisy_signal, label='Noisy Signal')
    plt.plot(denoised_signal, label='Denoised Signal')
    plt.legend()
    plt.title('Noise Removal Results')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

# Test the model on a sample
i = 0
plot_results(clean_signals[i], noisy_signals[i], autoencoder.predict(np.array([noisy_signals[i]]))[0])