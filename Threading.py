import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import cProfile
import pstats
from io import StringIO


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


def moving_average_parallel(data, spread, n_workers):
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
    data = np.random.rand(30000000)
    spread = 1000
    pr = cProfile.Profile()
    print("_" * 60, "Begin Averaging", "_" * 60)
    pr.enable()
    result = moving_average_parallel(data, spread, 4)
    pr.disable()
    print("=" * 60, "Averaging Finished", "=" * 60)

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats()

    print("Total time:", ps.total_tt)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(len(data)), data)
    ax2.plot(range(len(result)), result)
    ax2.set_ylim(0, 1)

    # plt.show()
