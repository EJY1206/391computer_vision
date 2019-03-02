import cv2
import numpy as np


def match_img(img1, img2, n=None):
    sift = cv2.xfeatures2d.SIFT_create()
    bfmatcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # use Harris corners for image matching
    if n == 1:
        corners = cv2.cornerHarris(gray1, 3, 3, 0.01)
        kpsCorners = np.argwhere(corners > 0.01 * corners.max())
        kpsCorners = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kpsCorners]

        cornersTwo = cv2.cornerHarris(gray2, 3, 3, 0.01)
        kpsCornersTwo = np.argwhere(cornersTwo > 0.01 * cornersTwo.max())
        kpsCornersTwo = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kpsCornersTwo]

        kpsCorners, dscCorners = sift.compute(gray1, kpsCorners)
        kpsCornersTwo, dscCornersTwo = sift.compute(gray2, kpsCornersTwo)

        matchesCorners = bfmatcher.match(dscCorners, dscCornersTwo)
        matchesCorners = sorted(matchesCorners, key=lambda x: x.distance)
        output = cv2.drawMatches(img1, kpsCorners, img2, kpsCornersTwo, matchesCorners[:15], None, flags=2)

    # use SIFT keypoints for image matching
    else:
        kp = sift.detect(gray1, None)
        kp, dsc = sift.compute(gray1, kp)

        kpTwo = sift.detect(gray2, None)
        kpTwo, dscTwo = sift.compute(gray2, kpTwo)

        matchesSift = bfmatcher.match(dsc, dscTwo)
        matchesSift = sorted(matchesSift, key=lambda x: x.distance)
        output = cv2.drawMatches(img1, kp, img2, kpTwo, matchesSift[:15], None, flags=2)

    return output


img = cv2.imread('puppy.jpg')
img = cv2.resize(img, (0,0), fx=0.4, fy=0.4)

# the scaled image
img_scale = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

num_rows, num_cols = img_scale.shape[:2]
translation_matrix = np.float32([[1,0,int(0.5*num_cols)],
[0,1,int(0.5*num_rows)] ])
rotation_matrix = cv2.getRotationMatrix2D((num_cols, num_rows), 30, 1)

# the translated image
img_translation = cv2.warpAffine(img_scale, translation_matrix, (2*num_cols,
2*num_rows))

# the rotated image
img_rotation = cv2.warpAffine(img_translation, rotation_matrix,
(num_cols*2, num_rows*2))


# the corner match scaling and rotation
img3 = match_img(img, img_scale, 1)
img4 = match_img(img, img_rotation, 1)

cv2.imshow('scaling.jpg', img3)
cv2.imshow('rotation.jpg', img4)

cv2.waitKey(0)
cv2.destroyAllWindows()
