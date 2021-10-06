import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

def linearSampler(x, y, image):
    x0 = math.floor(x)
    y0 = math.floor(y)
    a = x - x0
    b = y - y0
    ret = 0
    ret += (1 - a) * (1 - b) * image[y0, x0]
    ret += (a) * (1 - b) * image[y0, x0 + 1]
    ret += (1 - a) * (b) * image[y0 + 1, x0]
    ret += (a) * (b) * image[y0 + 1, x0 + 1]
    return ret

def nearestSampler(x, y, image):
    x0 = round(x)
    y0 = round(y)
    ret = image[y0, x0]
    return ret

image = np.array(Image.open("2021-09-23-TEM_sample_006(x80).tif")).astype(np.float32)
transform = np.fft.fftshift(np.fft.fft2(image))

r = 60
l = 125
m = -0.39
dt = 0.1
pxsize = 620.1172e-9
xmin = transform.shape[0] // 2 - r
xmax = transform.shape[0] // 2 + r
ymin = transform.shape[1] // 2 - r
ymax = transform.shape[1] // 2 + r
k0x = 2 * np.pi / (2560 * pxsize)
k0y = 2 * np.pi / (1920 * pxsize)
plt.imshow(np.log(np.abs(transform)**2), extent=((-2560/2-0.5) * k0x, (2560/2-0.5) * k0x, (-1920/2-0.5) * k0y, (1920/2-0.5) * k0y), vmax=35)
plt.xlabel("$k_x$ [$m^{-1}$]")
plt.ylabel("$k_y$ [$m^{-1}$]")
plt.show()
plt.imshow(np.log(np.abs(transform[xmin:xmax, ymin:ymax])**2), aspect=k0y/k0x, extent=(-r-0.5, r-0.5, -r-0.5, r-0.5), vmax=35)
t1 = np.arange(-l/2, l/2, dt)
nx1 = 1 / np.sqrt(1 + m**2)
ny1 = m / np.sqrt(1 + m**2)
dk1 = np.sqrt(nx1**2 * k0x**2 + ny1**2 * k0y**2)
plt.plot(nx1*t1, -ny1*t1, "r")
m = -1 / m * (k0x/k0y)**2
nx2 = 1 / np.sqrt(1 + m**2)
ny2 = m / np.sqrt(1 + m**2)
dk2 = np.sqrt(nx2**2 * k0x**2 + ny2**2 * k0y**2)
t2 = np.arange(-l/2, l/2, dt) / dk2 * dk1
plt.plot(nx2*t2, -ny2*t2, "b")
plt.xlabel("$k_x$ [pixel]")
plt.ylabel("$k_y$ [pixel]")
plt.show()

i1 = t1 * 0
i2 = t2 * 0
middle = np.abs(transform[xmin:xmax, ymin:ymax])**2
for currt1, currt2, n in zip(t1, t2, range(len(t1))):
    i1[n] = linearSampler(nx1 * currt1 + r, ny1 * currt1 + r, middle)
    i2[n] = linearSampler(nx2 * currt2 + r, ny2 * currt2 + r, middle)
k1 = dk1 * t1
k2 = dk2 * t2
plt.plot(k1, np.log(i1), "r")
plt.plot(k2, np.log(i2), "b")
plt.grid()
plt.xlabel("$k$ [$m^{-1}$]")
plt.ylabel("$\log(PSD)$")
i = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])
k0 = 49.3e3
plt.vlines(k0 * i, np.min(np.append(np.log(i1), np.log(i2))), np.max(np.append(np.log(i1), np.log(i2))), colors="orange")
plt.show()

print(2*np.pi/k0*1e6)