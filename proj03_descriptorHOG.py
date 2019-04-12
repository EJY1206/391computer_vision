import matplotlib.pyplot as plt
import cv2

from skimage.feature import hog
from skimage import data, exposure

# read images of palms and not palms
palm1 = cv2.imread("palm_1.png")
palm2 = cv2.imread("palm_2.png")
palm3 = cv2.imread("palm_3.png")
palm4 = cv2.imread("palm_4.png")
palm5 = cv2.imread("palm_5.png")
palm6 = cv2.imread("palm_6.png")
palm7 = cv2.imread("palm_7.png")
palm8 = cv2.imread("palm_8.png")
palm9 = cv2.imread("palm_9.png")
palm10 = cv2.imread("palm_10.png")

agriculture1 = cv2.imread("agriculture_1.png")
forest1 = cv2.imread("forest_1.png")
forest2 = cv2.imread("forest_2.png")
forest3 = cv2.imread("forest_3.png")
forest4 = cv2.imread("forest_4.png")
pond1 = cv2.imread('pond_1.png')
pond2 = cv2.imread('pond_2.png')
pond3 = cv2.imread('pond_3.png')
river1 = cv2.imread("river_1.png")
river2 = cv2.imread("river_2.png")

# image = data.coffee()
image = agriculture1

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

print(fd.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')

plt.figure(2)
plt.plot(fd)


plt.show()
