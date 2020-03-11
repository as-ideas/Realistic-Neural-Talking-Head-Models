from scipy.linalg import sqrtm
import numpy as np
import functools
import time
from pathlib import Path
import inspect


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        if float(run_time) > 1:
            filename = Path(inspect.stack()[1][1]).name
            print(
                f"Finished {func.__qualname__!r} in {filename} in {run_time:.4f} secs"
            )
        return value

    return wrapper_timer


def calculate_fid(x, x_hat, inception):
    x_out = inception(x)
    x_hat_out = inception(x_hat)

    mu1 = x_out.mean(axis=0)
    mu2 = x_hat_out.mean(axis=0)
    diff = mu1 - mu2
    sigma1 = np.cov(x_out.detach().numpy(), rowvar=False)
    sigma2 = np.cov(x_hat_out.detach().numpy(), rowvar=False)
    # calculate sqrt of product between cov
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
