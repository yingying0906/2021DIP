import cv2
import math
import numpy as np


def invert_fft(input):
    ishift = np.fft.ifftshift(input)
    ifft_2D = np.fft.ifft2(ishift)
    output = np.abs(ifft_2D)
    return output


def lowpass_filter(input, radius):
    r, c = input.shape
    output = np.zeros(input.shape)
    for i in range(r):
        for j in range(c):
            temp = ((i - r/2)**2 + (j-c/2)**2)**(1/2)
            if(temp <= radius):
                output[i][j] = 1
            else:
                output[i][j] = 0
    return output


def highpass_filter(input, radius):
    r, c = input.shape
    output = np.zeros(input.shape)
    for i in range(r):
        for j in range(c):
            temp = ((i - r/2)**2 + (j-c/2)**2)**(1/2)
            if(temp <= radius):
                output[i][j] = 0
            else:
                output[i][j] = 1
    return output


# 2a FFT
sample2 = cv2.imread("sample2.png", cv2.IMREAD_GRAYSCALE)
sample2 = sample2.astype(np.float32)
fft_2D = np.fft.fft2(sample2)
fft_shift = np.fft.fftshift(fft_2D)
result5 = np.log(np.abs(fft_shift))

# normalize
result5 = cv2.UMat(result5)
cv2.normalize(result5, result5, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("result5.png", result5)

# 2b frequency
lowpass = fft_shift * lowpass_filter(fft_shift, 30)
result6 = invert_fft(lowpass)
cv2.imwrite("result6_30.png", result6)

lowpass = fft_shift * lowpass_filter(fft_shift, 65)
result6 = invert_fft(lowpass)
cv2.imwrite("result6_65.png", result6)
cv2.imwrite("result6.png", result6)

# 2b pixel
low_pass_kernel = np.ones((5, 5), np.float32)/25
result7 = cv2.filter2D(sample2, -1, low_pass_kernel)
cv2.imwrite("result7.png", result7)

# 2c freq
highpass = fft_shift * highpass_filter(fft_shift, 15)
result8 = invert_fft(highpass)
cv2.imwrite("result8_15.png", result8)

highpass = fft_shift * highpass_filter(fft_shift, 80)
result8 = invert_fft(highpass)
cv2.imwrite("result8_80.png", result8)
cv2.imwrite("result8.png", result8)

# 2c pixel
high_pass_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
result9 = cv2.filter2D(sample2, -1, high_pass_kernel) - sample2
cv2.imwrite("result9.png", result9)

# 2d
sample3 = cv2.imread("sample3.png", cv2.IMREAD_GRAYSCALE)
sample3 = sample3.astype(np.float32)
result10_fft_2D = np.fft.fft2(sample3)
result10_fft_shift = np.fft.fftshift(result10_fft_2D)
result10 = np.log(np.abs(result10_fft_shift))

# normalize
result10 = cv2.UMat(result10)
cv2.normalize(result10, result10, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("result10.png", result10)

# 2e
filter = np.ones(sample3.shape, np.uint8)
filter[240][180] = 0
filter[240][460] = 0
result11 = result10_fft_shift*filter
result11 = invert_fft(result11)
cv2.imwrite("result11.png", result11)
