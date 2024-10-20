import tensorflow as tf
import numpy as np

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.convert_to_tensor(angle_rads[np.newaxis, ...], dtype=tf.float32)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        return self.dense(concat_attention)

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_shape):
        super(Transformer, self).__init__()
        self.embedding = tf.keras.layers.Dense(d_model)
        self.pos_encoding = positional_encoding(input_shape, d_model)
        
        self.encoder_layers = [MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
        self.ffn_layers = [point_wise_feed_forward_network(d_model, dff) for _ in range(num_layers)]
        
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = tf.keras.layers.Dense(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        
        for i in range(len(self.encoder_layers)):
            attn_output = self.encoder_layers[i](x, x, x)
            x = self.layernorm(x + attn_output)
            ffn_output = self.ffn_layers[i](x)
            x = self.layernorm(x + ffn_output)

        return self.output_layer(x)
    # def call(self, x):
    #     seq_len = tf.shape(x)[1]  # Get actual sequence length from input tensor
    #     x = self.embedding(x)  # Shape: (batch_size, seq_len, d_model)
        
    #     # Add positional encoding: Slice positional encoding to match the input sequence length
    #     x += self.pos_encoding[:, :seq_len, :]  # Shape: (batch_size, seq_len, d_model)
        
    #     for i in range(len(self.encoder_layers)):
    #         attn_output = self.encoder_layers[i](x, x, x)
    #         x = self.layernorm(x + attn_output)
    #         ffn_output = self.ffn_layers[i](x)
    #         x = self.layernorm(x + ffn_output)

    #     return self.output_layer(x)

# Example usage
# model = Transformer(num_layers=4, d_model=128, num_heads=8, dff=512, input_shape=100)
# input_signal = tf.random.normal([1, 100, 1])  # Example batch of noisy signals
# output_signal = model(input_signal)

import numpy as np

# Generate clean signal (e.g., sine wave)
def generate_clean_signal(length, freq=5):
    t = np.linspace(0, 1, length)
    return np.sin(2 * np.pi * freq * t)

# Add noise to generate noisy signal
def add_noise(clean_signal, noise_factor=0.5):
    noise = noise_factor * np.random.randn(*clean_signal.shape)
    return clean_signal + noise

# Create dataset
length = 100
clean_signals = np.array([generate_clean_signal(length) for _ in range(1000)])  # 1000 examples
noisy_signals = np.array([add_noise(sig) for sig in clean_signals])

padded_noisy_signals = tf.keras.preprocessing.sequence.pad_sequences(noisy_signals, maxlen=length, padding='post')
padded_clean_signals = tf.keras.preprocessing.sequence.pad_sequences(clean_signals, maxlen=length, padding='post')

loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model = Transformer(num_layers=4, d_model=100, num_heads=8, dff=512, input_shape=length)
model.compile(optimizer=optimizer, loss=loss_function)

# history = model.fit(padded_noisy_signals, padded_clean_signals, 
#                     epochs=50, 
#                     batch_size=100, 
#                     validation_split=0.1)

history = model.fit(noisy_signals, clean_signals, 
                    epochs=50, 
                    batch_size=20, 
                    validation_split=0.1)