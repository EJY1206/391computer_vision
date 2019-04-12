from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import numpy as np
import matplotlib.pyplot as plt
import cv2

# settings for LBP
radius = 3
n_points = 8 * radius
METHOD = 'uniform'


def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')

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

# convert the images to gray
palm1_gray = cv2.cvtColor(palm1, cv2.COLOR_RGB2GRAY)
palm2_gray = cv2.cvtColor(palm2, cv2.COLOR_RGB2GRAY)
palm3_gray = cv2.cvtColor(palm3, cv2.COLOR_RGB2GRAY)
palm4_gray = cv2.cvtColor(palm4, cv2.COLOR_RGB2GRAY)
palm5_gray = cv2.cvtColor(palm5, cv2.COLOR_RGB2GRAY)
palm6_gray = cv2.cvtColor(palm6, cv2.COLOR_RGB2GRAY)
palm7_gray = cv2.cvtColor(palm7, cv2.COLOR_RGB2GRAY)
palm8_gray = cv2.cvtColor(palm8, cv2.COLOR_RGB2GRAY)
palm9_gray = cv2.cvtColor(palm9, cv2.COLOR_RGB2GRAY)
palm10_gray = cv2.cvtColor(palm10, cv2.COLOR_RGB2GRAY)

agriculture1_gray = cv2.cvtColor(agriculture1, cv2.COLOR_RGB2GRAY)
forest1_gray = cv2.cvtColor(forest1, cv2.COLOR_RGB2GRAY)
forest2_gray = cv2.cvtColor(forest2, cv2.COLOR_RGB2GRAY)
forest3_gray = cv2.cvtColor(forest3, cv2.COLOR_RGB2GRAY)
forest4_gray = cv2.cvtColor(forest4, cv2.COLOR_RGB2GRAY)
pond1_gray = cv2.cvtColor(pond1, cv2.COLOR_RGB2GRAY)
pond2_gray = cv2.cvtColor(pond2, cv2.COLOR_RGB2GRAY)
pond3_gray = cv2.cvtColor(pond3, cv2.COLOR_RGB2GRAY)
river1_gray = cv2.cvtColor(river1, cv2.COLOR_RGB2GRAY)
river2_gray = cv2.cvtColor(river2, cv2.COLOR_RGB2GRAY)

image = agriculture1_gray

# image = data.load('brick.png')
lbp = local_binary_pattern(image, n_points, radius, METHOD)


def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')


# plot histograms of LBP of textures
fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
plt.gray()

titles = ('edge', 'flat', 'corner')
w = width = radius - 1
edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
i_14 = n_points // 4            # 1/4th of the histogram
i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                 list(range(i_34 - w, i_34 + w + 1)))

label_sets = (edge_labels, flat_labels, corner_labels)

for ax, labels in zip(ax_img, label_sets):
    ax.imshow(overlay_labels(image, lbp, labels))

for ax, labels, name in zip(ax_hist, label_sets, titles):
    counts, _, bars = hist(ax, lbp)
    highlight_bars(bars, labels)
    ax.set_ylim(top=np.max(counts[:-1]))
    ax.set_xlim(right=n_points + 2)
    ax.set_title(name)

ax_hist[0].set_ylabel('Percentage')
for ax in ax_img:
    ax.axis('off')


plt.show()
