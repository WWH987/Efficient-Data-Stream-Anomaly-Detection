from keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


def generate_data_stream(size=1000, noise_level=0.1):
    np.random.seed(42)
    t = np.arange(0, size)
    seasonal = np.sin(0.02 * t)
    trend = 0.01 * t
    noise = np.random.normal(0, noise_level, size)
    data_stream = seasonal + trend + noise
    return data_stream


def build_autoencoder(input_shape):
    encoder = models.Sequential([
        layers.InputLayer(input_shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu')
    ])

    decoder = models.Sequential([
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_shape, activation='sigmoid')
    ])

    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


def run_ae_algorithm():
    data_stream = generate_data_stream(size=1000, noise_level=0.2)
    autoencoder = build_autoencoder(1)

    # train encoder
    train_autoencoder(autoencoder, data_stream)

    # detect anomalies
    anomalies, reconstruction_error = detect_anomalies_with_autoencoder(autoencoder, data_stream)

    # draw result
    plt.figure()
    plt.plot(data_stream, label='Data Stream')
    plt.scatter(anomalies, data_stream[anomalies], color='red', label='Anomalies')
    plt.legend()

    # save img 'ae_output.png'
    plt.savefig('ae_output.png')
    plt.close()


def train_autoencoder(autoencoder, data_stream, epochs=20):
    normal_data = data_stream[:500].reshape(-1, 1)
    autoencoder.fit(normal_data, normal_data, epochs=epochs, batch_size=32, shuffle=True)


def detect_anomalies_with_autoencoder(autoencoder, data_stream, threshold=0.01):
    reconstructed_data = autoencoder.predict(data_stream.reshape(-1, 1))
    reconstruction_error = np.mean(np.abs(reconstructed_data - data_stream.reshape(-1, 1)), axis=1)

    anomalies = np.where(reconstruction_error > threshold)[0]
    return anomalies, reconstruction_error
