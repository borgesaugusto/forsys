from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

import matplotlib.pyplot as plt

image = io.imread("real_images/prim1n5.png")

grayscale = rgb2gray(image)

thresh = threshold_otsu(grayscale)

binary = image > thresh

# thresh = 

plt.imshow(grayscale, cmap=plt.cm.gray)
plt.axis("off")

plt.savefig("binary.png", dpi=300)
plt.clf()

skeleton = skeletonize(binary)
plt.imshow(skeleton, cmap=plt.cm.gray)
plt.axis("off")

plt.savefig("skeleton.png", dpi=300)