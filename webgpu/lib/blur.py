import numpy as np
from scipy.stats import norm

from kernel import kernel


def gaussian_kernel1d(n, sigma):
  total = (2 * norm.cdf(n - 0.5, scale=sigma) - 1)
  coefs = [(norm.cdf(i + 0.5, scale=sigma) - norm.cdf(i - 0.5, scale=sigma)) / total for i in range(n)]

  return np.array(coefs[:0:-1] + coefs)

def gaussian_kernel2d(n, sigma):
  kernel1d = gaussian_kernel1d(n, sigma)
  return np.outer(kernel1d, kernel1d)

  # return np.array((n - 1) * [0] + [1] + (n - 1) * [0])[:, np.newaxis] @ kernel1d[np.newaxis, :]
  return kernel1d[:, np.newaxis] @ kernel1d[np.newaxis, :]


def blur(runner, input_buffer, output_buffer, *, input_size, kernel_size = 3, sigma = 1):
  kernel_half_size = round((kernel_size + 1) * 0.5)
  return kernel(runner, input_buffer, output_buffer, input_size, gaussian_kernel2d(kernel_half_size, sigma))
