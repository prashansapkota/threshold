import cv2
import numpy as np
from scipy import ndimage
from PIL import Image

# Load image with PIL for cropping
image_path = '/Users/prashansapkota/Documents/7-1-24 LT 5 hr 90kx15.tif'
im_pil = Image.open(image_path)

# Crop the image (left, top, right, bottom)
width, height = im_pil.size
crop_area = (0, 0, width, height - 100)
im_pil = im_pil.crop(crop_area)

# Convert to OpenCV format (numpy array, BGR)
im = np.array(im_pil)
if len(im.shape) == 2:
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
else:
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

cv2.namedWindow('Filtered Particles')
cv2.namedWindow('Original Image')

# --- Sliders: both start at 0 ---
cv2.createTrackbar('Block Size', 'Filtered Particles', 0, 10, lambda x: None)  # block_size: 3 to 51
cv2.createTrackbar('C', 'Filtered Particles', 0, 40, lambda x: None)           # C: -20 to +20

while True:
    block_slider = cv2.getTrackbarPos('Block Size', 'Filtered Particles')
    c_slider = cv2.getTrackbarPos('C', 'Filtered Particles')

    block_size = 2 * block_slider + 3      # block_size always odd, starts at 3
    C = c_slider - 20                      # C goes from -20 to +20

    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

    labelarray, num_features = ndimage.label(cleaned)
    sizes = ndimage.sum(cleaned, labelarray, range(1, num_features + 1))

    min_area = 20
    mask = np.zeros_like(cleaned)

    for i, size in enumerate(sizes):
        if size > min_area:
            mask[labelarray == i + 1] = 255

    label_filtered, final_particle_count = ndimage.label(mask)

    display = im.copy()
    centroids = ndimage.center_of_mass(mask, label_filtered, range(1, final_particle_count + 1))

    for center in centroids:
        if not np.isnan(center[0]) and not np.isnan(center[1]):
            cv2.circle(display, (int(center[1]), int(center[0])), 5, (0, 255, 0), 2)

    text = f'Particles Detected: {final_particle_count}'
    cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Filtered Particles', display)
    cv2.imshow('Original Image', gray)  # Show original grayscale

    print(f'Particles: {final_particle_count}, Block Size: {block_size}, C: {C}', end='\r')

    key = cv2.waitKey(100)
    if key == 27:
        break

cv2.destroyAllWindows()

