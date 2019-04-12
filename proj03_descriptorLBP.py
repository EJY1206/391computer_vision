import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import cv2

plt.rcParams['font.size'] = 9

# settings for LBP, for more info see
# http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
#
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

# lpb is the local binary pattern computed for each pixel in the image
def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

# The Kullback-Leibler divergence is a measure of how one probability distribution
# is different from a second, reference probability distribution.
# These probability distributions are the histograms computed from the LBP
# KL(p,q) = 0 means p and q distributions are identical.
def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

# refs is an array reference LB patterns for various classes (brick, grass, wall)
# img is an input image
# match() gives the best match by comparing the KL divergence between the histogram
# of the img LBP and the histograms of the refs LBPs.
def match(refs, img):
    best_score = 10
    best_name = None
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, density=True, bins=n_bins,
                                   range=(0, n_bins))
        score = kullback_leibler_divergence(hist, ref_hist)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


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


refs = {
    'Palm': local_binary_pattern(palm2_gray, n_points, radius, METHOD),
    'Not Palm': local_binary_pattern(forest2_gray, n_points, radius, METHOD),
}

# classify the images, test if they are palms or not
print('Identify \'Palm\' or \'Not Palm\' images matched against references using LBP:')

# Expected Results: Palm
print('original: Palm, match result: ',
      match(refs, palm1_gray))
print('original: Palm, match result: ',
      match(refs, palm3_gray))
print('original: Palm, match result: ',
      match(refs, palm4_gray))
print('original: Palm, match result: ',
      match(refs, palm5_gray))

print('original: Palm, match result: ',
      match(refs, palm6_gray))
print('original: Palm, match result: ',
      match(refs, palm7_gray))
print('original: Palm, match result: ',
      match(refs, palm8_gray))
print('original: Palm, match result: ',
      match(refs, palm9_gray))
print('original: Palm, match result: ',
      match(refs, palm10_gray))

# Expected Results: Not Palm
print('original: Forest, match result: ',
      match(refs, forest1_gray))
print('original: Forest, match result: ',
      match(refs, forest3_gray))
print('original: Pond, match result: ',
      match(refs, pond1_gray))
print('original: Pond, match result: ',
      match(refs, river1_gray))
print('original: Agriculture, match result: ',
      match(refs, agriculture1_gray))
print('original: Forest, match result: ',
      match(refs, forest4_gray))
print('original: Pond, match result: ',
      match(refs, pond2_gray))
print('original: Pond, match result: ',
      match(refs, pond3_gray))
print('original: River, match result: ',
      match(refs, river2_gray))
