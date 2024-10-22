import random
import tensorflow as tf
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.get_positional_encoding(seq_len, d_model)
    
    def get_positional_encoding(self, seq_len, d_model, n=10000):
        # Create position array (shape: [seq_len, 1])
        position = np.arange(seq_len)[:, np.newaxis]

        # Create dimension array (shape: [d_model/2])
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(n) / d_model))

        # Compute the sinusoidal functions and stack them
        P = np.zeros((seq_len, d_model))
        P[:, 0::2] = np.sin(position * div_term)  # Apply sin to even indices
        P[:, 1::2] = np.cos(position * div_term)  # Apply cos to odd indices

        # Convert to TensorFlow tensor and reshape to (1, seq_len, d_model)
        P_tensor = tf.convert_to_tensor(P, dtype=tf.float32)
        P_tensor_reshaped = tf.reshape(P_tensor, (1, seq_len, d_model))

        return P_tensor_reshaped

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    # dff = dimension of feed forward network (size of the hidden layer)
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

class NoiseRemovalTransformer(tf.keras.Model):
    def __init__(self, d_model, seq_len, num_heads, dff, num_layers, dropout_rate=0.1):
        super(NoiseRemovalTransformer, self).__init__()
        # Embedding layer
        self.embedding_layer = tf.keras.layers.Dense(d_model, activation=None)

        # Positional encoding layer
        self.pos_encoding_layer = PositionalEncoding(seq_len, d_model)

        # Encoder layers
        self.enc_layers = [EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate) 
                           for _ in range(num_layers)]
        
        # Final layer
        self.final_layer = tf.keras.layers.Dense(1, activation='tanh') # Output a single value for each time step

    def call(self, x, training=None):
        x = self.embedding_layer(x)

        x = self.pos_encoding_layer(x)

        for i in range(len(self.enc_layers)):
            x = self.enc_layers[i](x)

        x = self.final_layer(x)

        return x

# d_model = 32
# num_heads = 4
# num_layers = 1
# dff = 256
# seq_len = 100

# # Linear embedding layer
# embedding_layer = tf.keras.layers.Dense(d_model, activation=None)
# embedding_layer.build(input_shape=(None, seq_len, 1))

# # Positional encoding layer
# pos_encoding_layer = PositionalEncoding(seq_len=seq_len, d_model=d_model)

# # Encoder layer
# encoder_layer = EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff)

# # Final layer
# final_layer = tf.keras.layers.Dense(1, activation='tanh')
# final_layer.build(input_shape=(None, seq_len, d_model))

# # Example input
# x = np.arange(300)
# x = x.reshape(3, 100, 1)
# x = tf.convert_to_tensor(x, dtype=tf.float32)

# # Apply the layers
# x = embedding_layer.call(x)  # Apply embedding layer
# x = pos_encoding_layer.call(x)  # Apply positional encoding layer
# x = encoder_layer.call(x)  # Apply encoder layer

# MARK: Training

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
        # noisy_signal = add_noise(clean_signal, base_noise_level, variable_noise_factor)
        noisy_signal = add_noise(clean_signal, np.random.uniform(0.1,0.5), np.random.uniform(0.3,0.7))
        
        clean_signals.append(clean_signal)
        noisy_signals.append(noisy_signal)
    
    return np.array(clean_signals), np.array(noisy_signals)


# Model parameters
d_model = 32
num_heads = 2
num_layers = 1
dff = 256
seq_len = 1000
dropout_rate = 0.1

# Generate training data
num_samples = 1000
base_noise_level = 0.3

clean_signals, noisy_signals = generate_dataset(num_samples, seq_len, base_noise_level=base_noise_level)

# Reshape the data for the model input
clean_signals = clean_signals.reshape(num_samples, seq_len, 1)
noisy_signals = noisy_signals.reshape(num_samples, seq_len, 1)

# Split the data into training and validation sets
train_split = 0.8
split_index = int(num_samples * train_split)

train_clean = clean_signals[:split_index]
train_noisy = noisy_signals[:split_index]
val_clean = clean_signals[split_index:]
val_noisy = noisy_signals[split_index:]

# # Define the model (assuming you've already defined the NoiseRemovalTransformer class)
# d_model = 128
# num_heads = 8
# dff = 512
# num_layers = 4

model = NoiseRemovalTransformer(
    d_model=d_model, 
    seq_len=seq_len, 
    num_heads=num_heads, 
    dff=dff, 
    num_layers=num_layers
)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# # Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)

# Train the model
history = model.fit(
    train_noisy, train_clean,
    validation_data=(val_noisy, val_clean),
    epochs=100,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
    shuffle=True
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
i = 0
plot_results(clean_signals[i], noisy_signals[i], model.predict(np.array([noisy_signals[i]]))[0])
