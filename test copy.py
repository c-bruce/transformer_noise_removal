import tensorflow as tf
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    if mask is not None:
        logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)

    return output

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output

class NoiseRemovalTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 maximum_position_encoding, rate=0.1):
        super(NoiseRemovalTransformer, self).__init__()

        self.d_model = d_model

        self.embedding = tf.keras.layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)

        self.enc_layers = [TransformerBlock(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        self.final_layer = tf.keras.layers.Dense(1)  # Output a single value for each time step

    def call(self, x, training=None):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding(x)

        x = self.dropout(x, training=training)

        for i in range(len(self.enc_layers)):
            x = self.enc_layers[i](x, training, mask=None)

        x = self.final_layer(x)

        return x

# Update TransformerBlock to have mask as an optional parameter
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

# Example usage
input_sequence_length = 1000
d_model = 128
num_heads = 8
dff = 512
num_layers = 4

model = NoiseRemovalTransformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=1,  # Since we're dealing with continuous signal, not discrete tokens
    maximum_position_encoding=input_sequence_length
)

# Example input (batch_size, time_steps, features)
# example_input = tf.random.normal((32, input_sequence_length, 1))
# output = model(example_input, training=False, mask=None)

# print(output.shape)  # Expected: (32, 1000, 1)

# MARK: Training
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# def generate_signal(length, freq=1, amplitude=1):
#     """Generate a simple sinusoidal signal."""
#     t = np.linspace(0, 1, length)
#     signal = amplitude * np.sin(2 * np.pi * freq * t)
#     return signal

# def add_noise(signal, noise_level=0.1):
#     """Add random noise to the signal."""
#     noise = np.random.normal(0, noise_level, len(signal))
#     noisy_signal = signal + noise
#     return noisy_signal

# def generate_dataset(num_samples, sequence_length, noise_level=0.1):
#     """Generate a dataset of clean and noisy signals."""
#     clean_signals = []
#     noisy_signals = []
    
#     for _ in range(num_samples):
#         freq = np.random.uniform(0.5, 5)  # Random frequency between 0.5 and 5 Hz
#         amplitude = np.random.uniform(0.5, 2)  # Random amplitude between 0.5 and 2
#         clean_signal = generate_signal(sequence_length, freq, amplitude)
#         noisy_signal = add_noise(clean_signal, noise_level)
        
#         clean_signals.append(clean_signal)
#         noisy_signals.append(noisy_signal)
    
#     return np.array(clean_signals), np.array(noisy_signals)
import random
import matplotlib.pyplot as plt
 
 
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

# Generate training data
num_samples = 1000
sequence_length = 100
noise_level = 0.1

clean_signals, noisy_signals = generate_dataset(num_samples, sequence_length, noise_level)

# Reshape the data for the model input
clean_signals = clean_signals.reshape(num_samples, sequence_length, 1)
noisy_signals = noisy_signals.reshape(num_samples, sequence_length, 1)

# Split the data into training and validation sets
train_split = 0.8
split_index = int(num_samples * train_split)

train_clean = clean_signals[:split_index]
train_noisy = noisy_signals[:split_index]
val_clean = clean_signals[split_index:]
val_noisy = noisy_signals[split_index:]

# Define the model (assuming you've already defined the NoiseRemovalTransformer class)
d_model = 128
num_heads = 8
dff = 512
num_layers = 4

model = NoiseRemovalTransformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=1,
    maximum_position_encoding=sequence_length
)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)

# Train the model
history = model.fit(
    train_noisy, train_clean,
    validation_data=(val_noisy, val_clean),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(122)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.show()

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
test_input = noisy_signals[0:1]  # Take the first noisy signal
denoised_signal = model.predict(test_input)

plot_results(clean_signals[0], noisy_signals[0], denoised_signal[0])
