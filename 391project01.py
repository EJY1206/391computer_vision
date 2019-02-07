import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from mpl_toolkits.mplot3d import Axes3D

# ================================= Part 3.1 =================================

# Read image of Ima and resize by 1/2
img = cv2.imread('puppy.jpg')
small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

cv2.imwrite('small_puppy.jpg', small)

size = 3
boxFilter = np.random.randn(size, size)
boxFilter = boxFilter/boxFilter.sum()
plt.matshow(boxFilter)
plt.show()

imgFilter = np.zeros(small.shape, np.float64)
imgFilter[:, :, 0] = cv2.filter2D(small[:, :, 0], -1, boxFilter)
imgFilter[:, :, 1] = cv2.filter2D(small[:, :, 1], -1, boxFilter)
imgFilter[:, :, 2] = cv2.filter2D(small[:, :, 2], -1, boxFilter)

cv2.imwrite('filtered_image.jpg', imgFilter)


# ================================= Part 3.2.1 =================================

# Create noisy image as g = f + sigma * noise
# with noise scaled by sigma = .2 max(f)/max(noise)
noise = np.random.randn(small.shape[0], small.shape[1])
smallNoisy = np.zeros(small.shape, np.float64)
sigma = 0.2 * small.max()/noise.max()
# Color images need noise added to all channels
if len(small.shape) == 2:
    smallNoisy = small + sigma * noise
else:
    smallNoisy[:, :, 0] = small[:, :, 0] + sigma * noise
    smallNoisy[:, :, 1] = small[:, :, 1] + sigma * noise
    smallNoisy[:, :, 2] = small[:, :, 2] + sigma * noise

# save the puppy noisy image
# cv2.imwrite('Ima-noisy.jpg', smallNoisy)

noisy = cv2.imread('Ima-noisy.jpg')

# Apply Gaussian filter with 3x3
gauss3 = cv2.GaussianBlur(noisy,(3,3),0)
# cv2.imwrite('gaussian_image_noisex3.jpg', gauss3)

# Apply Gaussian filter with 9x9
gauss9 = cv2.GaussianBlur(noisy,(9,9),0)
# cv2.imwrite('gaussian_image_9x9.jpg', gauss9)

# Apply Gaussian filter with 27x27
gauss27 = cv2.GaussianBlur(noisy,(27,27),0)
# cv2.imwrite('gaussian_image_field7x27.jpg', gauss27)

# Apply median filter with 3x3
median3 = cv2.medianBlur(noisy, 3)
# cv2.imwrite('median_image_noisex3.jpg', median3)

# Apply median filter with 9x9
median9 = cv2.medianBlur(noisy, 9)
# cv2.imwrite('median_image_9x9.jpg', median9)

# Apply median filter with 27x27
median27 = cv2.medianBlur(noisy, 27)
# cv2.imwrite('median_image_field7x27.jpg', median27)

# ================================= Part 3.2.2 =================================

# Apply Canny edge detector on the puppy image
edgeOrig = cv2.Canny(small,50,200)
cv2.imwrite('Edge_image_original.jpg', edgeOrig)

# Apply Canny edge detector on the noisy puppy image
edgeNoise = cv2.Canny(noisy,50,200)
cv2.imwrite('Edge_image_noisy.jpg', edgeNoise)

field = cv2.imread('window-02-03.jpg')
edgeField = cv2.Canny(field,50,200)
cv2.imwrite('Edge_image_field.jpg', edgeField)

# ================================= Part 4.1 =================================

# Change the color of image to gray
graySmall = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
cv2.imwrite('gray_image.jpg', graySmall)
#

# Take the 2-D DFT and plot the magnitude of the corresponding Fourier coefficients
F2_graySmall = np.fft.fft2(graySmall)
fig = plt.figure()
# ax = fig.gca(projection='3d')
Y = (np.linspace(-int(graySmall.shape[0]/2), int(graySmall.shape[0]/2)-1, graySmall.shape[0]))
X = (np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1]))
X, Y = np.meshgrid(X, Y)

# Plot the magnitude and the log(magnitude + 1) as images (view from the top)
magnitudeImage = np.fft.fftshift(np.abs(F2_graySmall))
magnitudeImage = magnitudeImage / magnitudeImage.max()   # scale to [0, 1]
magnitudeImage = ski.img_as_ubyte(magnitudeImage)
logMagnitudeImage = np.fft.fftshift(np.log(np.abs(F2_graySmall)+1))
logMagnitudeImage = logMagnitudeImage / logMagnitudeImage.max()   # scale to [0, 1]
logMagnitudeImage = ski.img_as_ubyte(logMagnitudeImage)
# cv2.imwrite('Magnitude_image.jpg', magnitudeImage)
# cv2.imwrite('Log_Magnitude_image.jpg', logMagnitudeImage)

# ================================= Part 4.2 =================================

# analyzing image content and the decay of the magnitude of the Fourier coefficients
image_field = cv2.imread("window-02-03.jpg")
image_noise = cv2.imread("Ima-noisy.jpg")
image_puppy = cv2.imread("small_puppy.jpg")

# 1-D Fourier
# F_image_field = np.fft.fft(image_field.astype(float))
# F_image_noise = np.fft.fft(image_noise.astype(float))

# image_field
col_image_field = int(image_field.shape[1]/2)
# Obtain the image data for this column
colData_image_field = image_field[0:image_field.shape[0], col_image_field, 0]
# 1-D Fourier
F_colData_image_field = np.fft.fft(colData_image_field.astype(float))

# image_noise
col_image_noise = int(image_noise.shape[1]/2)
# Obtain the image data for this column
colData_image_noise = image_noise[0:image_noise.shape[0], col_image_noise, 0]
# 1-D Fourier
F_colData_image_noise = np.fft.fft(colData_image_noise.astype(float))

# image_puppy
col_image_puppy = int(image_puppy.shape[1]/2)
# Obtain the image data for this column
colData_image_puppy = image_puppy[0:image_puppy.shape[0], col_image_puppy, 0]
# 1-D Fourier
F_colData_image_puppy = np.fft.fft(colData_image_puppy.astype(float))


# plot
# image_field
xvalues = np.linspace(0, len(colData_image_field), len(colData_image_field))
markerline, stemlines, baseline = plt.stem(xvalues, np.fft.fftshift(np.abs(F_colData_image_field)), 'g')
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','r', 'linewidth', 0.5)
# oneDim_mag_image_field = plt.setp
# oneDim_mag_image_field.savefig("oneDim_mag_image_field.jpg")
plt.title("1-D Fourier - image_field"), plt.show()

# image_noise
xvalues = np.linspace(0, len(colData_image_noise), len(colData_image_noise))
markerline, stemlines, baseline = plt.stem(xvalues, np.fft.fftshift(np.abs(F_colData_image_noise)), 'g')
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color', 'r', 'linewidth', 0.5)
# oneDim_mag_image_noise = plt.setp
# oneDim_mag_image_noise.savefig("oneDim_mag_image_noise.jpg")
plt.title("1-D Fourier - image_noise"), plt.show()

# image_puppy
xvalues = np.linspace(0, len(colData_image_puppy), len(colData_image_puppy))
markerline, stemlines, baseline = plt.stem(xvalues, np.fft.fftshift(np.abs(F_colData_image_puppy)), 'g')
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','r', 'linewidth', 0.5)
plt.title("1-D Fourier - image_puppy"), plt.show()

cv2.waitKey(0)

# ================================= Part 5 =================================

# Explore the Butterworth filter
# U and V are arrays that give all integer coordinates in the 2-D plane
#  [-m/2 , m/2] x [-n/2 , n/2].
# Use U and V to create 3-D functions over (U,V)
U = (np.linspace(-int(graySmall.shape[0]/2), int(graySmall.shape[0]/2)-1, graySmall.shape[0]))
V = (np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1]))
U, V = np.meshgrid(U, V)
# The function over (U,V) is distance between each point (u,v) to (0,0)
D = np.sqrt(X*X + Y*Y)
# create x-points for plotting
xval = np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1])
# Specify a frequency cutoff value as a function of D.max()
D0 = 0.1 * D.max()

# The ideal lowpass filter makes all D(u,v) where D(u,v) <= 0 equal to 1
# and all D(u,v) where D(u,v) > 0 equal to 0
idealLowPass = D <= D0

# Filter our small grayscale image with the ideal lowpass filter
# 1. DFT of image
FTgraySmall = np.fft.fft2(graySmall.astype(float))
# 2. Butterworth filter is already defined in Fourier space
# 3. Elementwise product in Fourier space (notice fftshift of the filter)
FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(idealLowPass)
# 4. Inverse DFT to take filtered image back to the spatial domain
graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))

# Save the filter and the filtered image (after scaling)
idealLowPass = ski.img_as_ubyte(idealLowPass / idealLowPass.max())
graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
highFiltered = graySmall - graySmallFiltered
cv2.imwrite("idealLowPass.jpg", idealLowPass)
cv2.imwrite("low-grayImageIdealLowpassFiltered.jpg", graySmallFiltered)
cv2.imwrite("high-grayImageIdealLowpassFiltered.jpg", highFiltered)


# Plot the ideal filter and then create and plot Butterworth filters of order
# n = 1, 2, 3, 4
plt.plot(xval, idealLowPass[int(idealLowPass.shape[0]/2), :], 'c--', label='ideal')
colors='brgkmc'
for n in range(1, 5):
    # Create Butterworth filter of order n
    H = 1.0 / (1 + (np.sqrt(2) - 1)*np.power(D/D0, 2*n))
    # Apply the filter to the grayscaled image
    FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(H)
    graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))
    graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
    highFiltered = graySmall - graySmallFiltered
    cv2.imwrite("low-grayImageButterworth-n" + str(n) + ".jpg", graySmallFiltered)
    cv2.imwrite("high-grayImageButterworth-n" + str(n) + ".jpg", highFiltered)

    H = ski.img_as_ubyte(H / H.max())
    cv2.imwrite("low-butter-n" + str(n) + ".jpg", H)
    # Get a slice through the center of the filter to plot in 2-D
    slice = H[int(H.shape[0]/2), :]
    plt.plot(xval, slice, colors[n-1], label='n='+str(n))
    plt.legend(loc='upper left')

# plt.savefig('butterworthFilters.jpg', bbox_inches='tight')

