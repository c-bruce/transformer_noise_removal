import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib import rc, rcParams

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 16})
rc('text', usetex=True)

def get_positional_encoding(seq_len, d_model, n=10000):
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

def plot_positional_encoding(pos_encoding, cmap='RdBu'):
    plt.figure(figsize=(16, 4.5))
    plt.pcolormesh(pos_encoding.T, cmap=cmap)
    plt.xlabel('Position Index')
    plt.ylabel('Embedding Dimension')
    plt.clim(-1, 1)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('positional_encoding.png', format='png', dpi=1200)
    plt.show()


positional_encoding = np.array(get_positional_encoding(100, 32))[0]
plot_positional_encoding(positional_encoding)