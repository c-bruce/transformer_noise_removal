import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib import rc

from generate_synthetic_data import generate_dataset

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.get_positional_encoding(seq_len, d_model)
    
    def get_positional_encoding(self, seq_len, d_model, n=10000):
        # Create position array (shape: [seq_len, 1])
        position = np.arange(seq_len)[:, np.newaxis]

        # Create dimension array (shape: [d_model/2])
        denominator = np.power(n, 2 * np.arange(0, d_model, 2) / d_model)

        # Compute the sinusoidal functions and stack them
        P = np.zeros((seq_len, d_model))
        P[:, 0::2] = np.sin(position / denominator)  # Apply sin to even indices
        P[:, 1::2] = np.cos(position / denominator)  # Apply cos to odd indices

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
    self.add = tf.keras.layers.Add()
    self.layernorm = tf.keras.layers.LayerNormalization()

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output, attn_scores = self.mha(
        query=x,
        value=x,
        key=x,
        return_attention_scores=True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x, attn_scores

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output, attn_scores = self.mha(
        query=x,
        value=x,
        key=x,
        return_attention_scores=True,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x, attn_scores

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
    x, attn_scores = self.self_attention(x)
    x = self.ffn(x)
    return x, attn_scores

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
        self.final_layer = tf.keras.layers.Dense(1, activation=None) # Output a single value for each time step

    def call(self, x, return_attn_scores=False):
        x = self.embedding_layer(x)

        x = self.pos_encoding_layer(x)

        for i in range(len(self.enc_layers)):
            x, attn_scores = self.enc_layers[i](x)

        x = self.final_layer(x)

        if return_attn_scores:
            return x, attn_scores
        
        return x

# MARK: Training

# Model parameters
d_model = 32
num_heads = 1
num_layers = 1
dff = 128
seq_len = 100
dropout_rate = 0.1

# Generate training data
num_samples = 1000

clean_signals, noisy_signals = generate_dataset(num_samples, seq_len)

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
    plt.figure(figsize=(12, 3))
    plt.plot(clean_signal, color='b', ls='--', label='Clean Signal')
    plt.plot(noisy_signal, color='r', lw=0.5, label='Noisy Signal')
    plt.plot(denoised_signal, color='b', label='Denoised Signal')
    # plt.legend()
    # plt.title('Noise Removal Results')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.tight_layout()
    ax = plt.gca()
    ax.set_facecolor('whitesmoke')
    plt.show()

def plot_attention_scores(attn_scores, cmap='cmr.cosmic'):
    plt.figure(figsize=(4, 4))
    plt.imshow(attn_scores, cmap=cmap)
    plt.title('Attention Scores')
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.colorbar()
    plt.show()

def plot_positional_encoding(pos_encoding, cmap='cmr.cosmic'):
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(pos_encoding.T, cmap=cmap)
    plt.xlabel('Position')
    plt.ylabel('Depth')
    plt.title('Positional Encoding')
    plt.colorbar()
    plt.show()

# Test the model on a sample
i = 0
plot_results(clean_signals[i], noisy_signals[i], model.predict(np.array([noisy_signals[i]]))[0])
