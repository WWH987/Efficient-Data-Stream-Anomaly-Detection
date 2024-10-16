import numpy as np
from scipy import stats
import matplotlib.pyplot as plt



def generate_data_stream(size=1000, noise_level=0.1):
    np.random.seed(42)
    # create a synthetic data stream
    t = np.arange(0, size)
    seasonal = np.sin(0.02 * t)  # simulate seasonal pattern
    trend = 0.01 * t  # upward trend
    noise = np.random.normal(0, noise_level, size)  # add random noise
    data_stream = seasonal + trend + noise

    return data_stream


def detect_anomalies(data, threshold=3):
    z_scores = stats.zscore(data)
    anomalies = np.where(np.abs(z_scores) > threshold)[0]
    return anomalies


current_index = 50


def run_z_score_algorithm():
    global current_index
    data_stream = generate_data_stream(size=1000, noise_level=0.2)

    # realtime simulation
    window_size = 50
    if current_index + window_size > len(data_stream):
        current_index = window_size
    window = data_stream[current_index - window_size:current_index]

    anomalies = detect_anomalies(window, threshold=3)

    # draw plot and save
    plt.clf()
    plt.plot(range(current_index - window_size, current_index), window)
    plt.scatter(anomalies + current_index - window_size, window[anomalies], color='red')
    plt.savefig('z_score_output.png')
    plt.close()

    # simulate scroll for the next page
    current_index += 10


