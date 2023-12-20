import numpy as np
from scipy.fft import fft, ifft
from kernalization import SampleGeneration
from kernalization import FunctionApprox
from kernalization import KernelInitialization
from scipy.stats import norm
import math
from cmath import sqrt
from cmath import exp
import matplotlib.pyplot as plt


x_arr = np.arange(-3, 3.01, 0.01)
# p_arr = np.exp(- x_arr ** 2)
p_arr = norm.pdf(x_arr)
leng = len(x_arr)
arr1 = p_arr[(leng // 2):]
arr2 = p_arr[:leng // 2]
a_arr = np.concatenate((arr1, arr2))
# a_arr = p_arr
# ifft_arr = np.fft.ifft(a_arr)
ifft = fft(a_arr)
ifft = np.fft.ifftshift(a_arr)
# for n in range(len(ifft_arr)):
    # ifft_arr[n] = ifft_arr[n] * (x_arr[1] - x_arr[0]) * exp(- (2 * math.pi * 1j * n * x_arr[0])/len(x_arr) * (x_arr[1] - x_arr[0]))

# ifft_arr1 = ifft_arr[leng // 2 + 1:]
# ifft_arr2 = ifft_arr[:leng // 2 + 1]
# ifft = np.concatenate((ifft_arr1, ifft_arr2))
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x_arr, p_arr, c='b', label='source')
ax1.plot(x_arr, ifft, c='r', label='target')
print(p_arr)
print(ifft)
plt.legend(loc='upper left')
plt.xlabel("x")
plt.ylabel("y")
# plt.show()

# Plotting the Gaussian PDF and FFT result
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.title('Gaussian Probability Density Function')
plt.plot(x_arr, p_arr)
plt.xlabel('x')
plt.ylabel('PDF')
plt.grid()

plt.subplot(122)
plt.title('FFT Result')
plt.plot(ifft.real)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.show()