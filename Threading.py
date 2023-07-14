import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


def worker(data, start, end, spread):
    cumsum = np.cumsum(data[start:end])
    avg_vals = np.empty(end - start)
    for i in range(start, end):
        if i >= spread:
            avg_vals[i - start] = (
                cumsum[i - start] - cumsum[i - start - spread]
            ) / spread
        else:
            avg_vals[i - start] = cumsum[i - start] / (i + 1)
    return avg_vals


def moving_average_parallel(data, spread):
    n_workers = 1
    chunk_size = len(data) // n_workers
    with ProcessPoolExecutor() as executor:
        results = executor.map(
            worker,
            [data] * n_workers,
            range(0, len(data), chunk_size),
            range(chunk_size, len(data) + chunk_size, chunk_size),
            [spread] * n_workers,
        )
    return np.concatenate(list(results))


if __name__ == "__main__":
    data = np.random.rand(450000)
    spread = 100
    result = moving_average_parallel(data, spread)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(len(data)), data)
    ax2.plot(range(len(result)), result)
    ax2.set_ylim(0, 1)

    plt.show()
