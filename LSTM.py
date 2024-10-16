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


def run_lstm_algorithm():
    data_stream = generate_data_stream(size=1000, noise_level=0.2)
    lstm_model = build_lstm_model(input_shape=(50, 1))

    # train model
    train_lstm_model(lstm_model, data_stream)

    # detect anomalies
    anomalies, reconstruction_error = detect_anomalies_with_lstm(lstm_model, data_stream)

    # draw result
    plt.figure()
    plt.plot(data_stream, label='Data Stream')
    plt.scatter(anomalies, data_stream[anomalies], color='red', label='Anomalies')
    plt.legend()

    # save img 'lstm_output.png'
    plt.savefig('lstm_output.png')
    plt.close()


def create_dataset(data, time_steps=50):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    model = models.Sequential()
    model.add(layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(layers.LSTM(32, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm_model(model, data_stream, time_steps=50, epochs=20):
    X, y = create_dataset(data_stream, time_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model.fit(X, y, epochs=epochs, batch_size=32, shuffle=True)


def detect_anomalies_with_lstm(model, data_stream, time_steps=50, threshold=0.01):
    X, y = create_dataset(data_stream, time_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    predicted = model.predict(X)
    reconstruction_error = np.abs(predicted - y)

    anomalies = np.where(reconstruction_error > threshold)[0]
    return anomalies, reconstruction_error
