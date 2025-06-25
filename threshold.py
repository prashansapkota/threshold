# import cv2
# import pylab
# from scipy import ndimage

# im = cv2.imread('/Users/prashansapkota/Downloads/7-1-24+LT+3+hr+90kx17.tif')
# pylab.figure(0)
# pylab.imshow(im)

# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5,5), 0)
# maxValue = 255
# adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C#cv2.ADAPTIVE_THRESH_MEAN_C #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
# thresholdType = cv2.THRESH_BINARY#cv2.THRESH_BINARY #cv2.THRESH_BINARY_INV
# blockSize = 17 #odd number like 3,5,7,9,11
# C = -15 # constant to be subtracted
# im_thresholded = cv2.adaptiveThreshold(gray, maxValue, adaptiveMethod, thresholdType, blockSize, C) 
# labelarray, particle_count = ndimage.measurements.label(im_thresholded)
# print ("No. of Particle: ", particle_count)
# pylab.figure(1)
# pylab.imshow(im_thresholded, cmap='gray')
# pylab.show()





# import cv2
# import pylab
# import numpy as np
# from scipy import ndimage

# # === Step 1: Load and Preprocess the Image ===
# image_path = '/Users/prashansapkota/Downloads/7-1-24+LT+3+hr+90kx17.tif'
# im = cv2.imread(image_path)

# # Convert to grayscale and blur to reduce noise
# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)

# # === Step 2: Adaptive Thresholding ===
# maxValue = 255
# adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
# thresholdType = cv2.THRESH_BINARY
# blockSize = 17  # should be odd
# C = -15  # subtract from mean/gaussian to get threshold

# im_thresholded = cv2.adaptiveThreshold(gray, maxValue, adaptiveMethod, thresholdType, blockSize, C)

# raw_labelarray, raw_particle_count = ndimage.label(im_thresholded)
# print("Raw particle count (before filtering):", raw_particle_count)

# # === Step 3: Morphological Opening to Remove Noise ===
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# cleaned = cv2.morphologyEx(im_thresholded, cv2.MORPH_OPEN, kernel)



# # === Step 4: Filter Small Objects by Area ===
# # Label connected components
# labelarray, num_features = ndimage.label(cleaned)

# # Compute area of each labeled component
# sizes = ndimage.sum(cleaned, labelarray, range(1, num_features + 1))
# min_area = 20  # Minimum area to be counted as a real particle
# mask = np.zeros_like(cleaned)

# # Rebuild image with only valid (large enough) particles
# for i, size in enumerate(sizes):
#     if size > min_area:
#         mask[labelarray == i + 1] = 255

# # Final labeling of cleaned image
# label_filtered, final_particle_count = ndimage.label(mask)

# print("Final particle count (after filtering):", final_particle_count)

# # Convert original image to color (in case it was not already)
# output_image = im.copy()

# # Compute centers of mass of each labeled region
# centroids = ndimage.center_of_mass(mask, label_filtered, range(1, final_particle_count + 1))

# # Draw green circles on each centroid
# for center in centroids:
#     if not np.isnan(center[0]) and not np.isnan(center[1]):
#         center_coordinates = (int(center[1]), int(center[0]))  # Note: (x, y) = (col, row)
#         cv2.circle(output_image, center_coordinates, radius=5, color=(0, 255, 0), thickness=2)  # Green color

# # === Step 5: Visualization ===
# pylab.figure(figsize=(14, 6))

# # pylab.subplot(1, 2, 1)
# # pylab.imshow(im)
# # pylab.title("Original Image")
# # pylab.axis('off')

# # pylab.subplot(1, 3, 2)
# pylab.imshow(im_thresholded, cmap='gray')
# pylab.title("Raw Adaptive Threshold")
# pylab.axis('off')

# # pylab.subplot(1, 2, 2)
# # pylab.imshow(mask, cmap='gray')
# # pylab.title("Adaptive local thresholding with morphological noise removal")
# # pylab.axis('off')

# # pylab.tight_layout()
# # pylab.show()


# pylab.subplot(1, 2, 1)
# pylab.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
# pylab.title("Original Image")
# pylab.axis('off')

# pylab.subplot(1, 2, 2)
# pylab.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
# pylab.title("Detected Particles with Green Circles")
# pylab.axis('off')

# pylab.tight_layout()
# pylab.show()









# import cv2
# import pylab
# import numpy as np
# from scipy import ndimage

# # Load and Preprocess the Image
# image_path = '/Users/prashansapkota/Downloads/7-1-24+LT+3+hr+90kx17.tif'
# im = cv2.imread(image_path)

# # Convert to grayscale and blur to reduce noise
# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)

# # Adaptive Thresholding
# im_thresholded = cv2.adaptiveThreshold(
#     gray,
#     maxValue=255,
#     adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     thresholdType=cv2.THRESH_BINARY,
#     blockSize=17,
#     C=-15
# )

# # Raw Particle Detection (Before Filtering)
# raw_labelarray, raw_particle_count = ndimage.label(im_thresholded)
# print("Raw particle count (before filtering):", raw_particle_count)
# raw_centroids = ndimage.center_of_mass(im_thresholded, raw_labelarray, range(1, raw_particle_count + 1))

# # Draw green circles on original image for raw detection
# raw_detect_image = im.copy()
# for center in raw_centroids:
#     if not np.isnan(center[0]) and not np.isnan(center[1]):
#         cv2.circle(raw_detect_image, (int(center[1]), int(center[0])), 5, (0, 255, 0), 2)

# # Morphological Opening and Size Filtering
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# cleaned = cv2.morphologyEx(im_thresholded, cv2.MORPH_OPEN, kernel)

# labelarray, num_features = ndimage.label(cleaned)
# sizes = ndimage.sum(cleaned, labelarray, range(1, num_features + 1))

# min_area = 20
# mask = np.zeros_like(cleaned)

# for i, size in enumerate(sizes):
#     if size > min_area:
#         mask[labelarray == i + 1] = 255

# label_filtered, final_particle_count = ndimage.label(mask)
# print("Final particle count (after filtering):", final_particle_count)

# # Draw Green Circles After Filtering 
# filtered_detect_image = im.copy()
# filtered_centroids = ndimage.center_of_mass(mask, label_filtered, range(1, final_particle_count + 1))

# for center in filtered_centroids:
#     if not np.isnan(center[0]) and not np.isnan(center[1]):
#         cv2.circle(filtered_detect_image, (int(center[1]), int(center[0])), 5, (0, 255, 0), 2)

# # Show Comparison
# pylab.figure(figsize=(14, 7))

# pylab.subplot(1, 2, 1)
# pylab.imshow(cv2.cvtColor(raw_detect_image, cv2.COLOR_BGR2RGB))
# pylab.title(f"Before Filtering: {raw_particle_count} Particles")
# pylab.axis('off')

# pylab.subplot(1, 2, 2)
# pylab.imshow(cv2.cvtColor(filtered_detect_image, cv2.COLOR_BGR2RGB))
# pylab.title(f"After Filtering: {final_particle_count} Particles")
# pylab.axis('off')

# pylab.tight_layout()
# pylab.show()







# import cv2
# import numpy as np
# from scipy import ndimage

# # Load image
# image_path = '/Users/prashansapkota/Downloads/7-1-24+LT+4+hr+90kx13.tif'
# im = cv2.imread(image_path)

# if im is None:
#     print("Failed to load image.")
#     exit()

# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)

# cv2.namedWindow('Filtered Particles')

# # Block Size slider: 0–13 maps to odd block sizes from 3 to 29
# cv2.createTrackbar('Block Size', 'Filtered Particles', 7, 13, lambda x: None)  
# # C slider: 0–40 maps to -20 to +20
# cv2.createTrackbar('C', 'Filtered Particles', 20, 40, lambda x: None)  

# while True:
#     # Get slider values
#     block_slider = cv2.getTrackbarPos('Block Size', 'Filtered Particles')
#     c_slider = cv2.getTrackbarPos('C', 'Filtered Particles')

#     # Map sliders to usable values
#     block_size = 2 * block_slider + 3  # 3 to 29, odd numbers only
#     C = c_slider - 20  # -20 to +20

#     # Adaptive Threshold
#     adaptive_thresh = cv2.adaptiveThreshold(
#         gray,
#         maxValue=255,
#         adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         thresholdType=cv2.THRESH_BINARY,
#         blockSize=block_size,
#         C=C
#     )

#     # Morphological cleaning
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

#     labelarray, num_features = ndimage.label(cleaned)
#     sizes = ndimage.sum(cleaned, labelarray, range(1, num_features + 1))

#     min_area = 20
#     mask = np.zeros_like(cleaned)

#     for i, size in enumerate(sizes):
#         if size > min_area:
#             mask[labelarray == i + 1] = 255

#     label_filtered, final_particle_count = ndimage.label(mask)

#     # Draw detections
#     display = im.copy()
#     centroids = ndimage.center_of_mass(mask, label_filtered, range(1, final_particle_count + 1))

#     for center in centroids:
#         if not np.isnan(center[0]) and not np.isnan(center[1]):
#             cv2.circle(display, (int(center[1]), int(center[0])), 5, (0, 255, 0), 2)

#     # Overlay text with particle count
#     text = f'Particles Detected: {final_particle_count}'
#     cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     cv2.imshow('Filtered Particles', display)
#     print(f'Particles: {final_particle_count}, Block Size: {block_size}, C: {C}', end='\r')

#     key = cv2.waitKey(100)
#     if key == 27:  # ESC to exit
#         break

# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from scipy import ndimage
# from PIL import Image

# # Load image with PIL for cropping
# image_path = '/Users/prashansapkota/Documents/7-1-24 LT 3 hr 90kx17.tif'
# im_pil = Image.open(image_path)

# # Crop the image (left, top, right, bottom)
# # Example: Remove 50 pixels from bottom — adjust as needed
# width, height = im_pil.size
# crop_area = (0, 0, width, height - 100)  
# im_pil = im_pil.crop(crop_area)

# # Convert back to OpenCV format (numpy array, BGR)
# im = np.array(im_pil)
# if len(im.shape) == 2:
#     im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
# else:
#     im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)

# cv2.namedWindow('Filtered Particles')

# # Sliders for block size and C
# cv2.createTrackbar('Block Size', 'Filtered Particles', 7, 13, lambda x: None)  
# cv2.createTrackbar('C', 'Filtered Particles', 20, 40, lambda x: None)  

# while True:
#     block_slider = cv2.getTrackbarPos('Block Size', 'Filtered Particles')
#     c_slider = cv2.getTrackbarPos('C', 'Filtered Particles')

#     block_size = 2 * block_slider + 3  
#     C = c_slider - 20  

#     adaptive_thresh = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
#     )

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

#     labelarray, num_features = ndimage.label(cleaned)
#     sizes = ndimage.sum(cleaned, labelarray, range(1, num_features + 1))

#     min_area = 20
#     mask = np.zeros_like(cleaned)

#     for i, size in enumerate(sizes):
#         if size > min_area:
#             mask[labelarray == i + 1] = 255

#     label_filtered, final_particle_count = ndimage.label(mask)

#     display = im.copy()
#     centroids = ndimage.center_of_mass(mask, label_filtered, range(1, final_particle_count + 1))

#     for center in centroids:
#         if not np.isnan(center[0]) and not np.isnan(center[1]):
#             cv2.circle(display, (int(center[1]), int(center[0])), 5, (0, 255, 0), 2)

#     text = f'Particles Detected: {final_particle_count}'
#     cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     cv2.imshow('Filtered Particles', display)
#     print(f'Particles: {final_particle_count}, Block Size: {block_size}, C: {C}', end='\r')

#     key = cv2.waitKey(100)
#     if key == 27:
#         break

# cv2.destroyAllWindows()









# import cv2
# import numpy as np
# from scipy import ndimage
# from PIL import Image

# # Load image with PIL for cropping
# image_path = '/Users/prashansapkota/Documents/7-1-24 LT 4 hr 90kx10.tif'
# im_pil = Image.open(image_path)

# # Crop the image (left, top, right, bottom)
# width, height = im_pil.size
# crop_area = (0, 0, width, height - 100)
# im_pil = im_pil.crop(crop_area)

# # Convert to OpenCV format (numpy array, BGR)
# im = np.array(im_pil)
# if len(im.shape) == 2:
#     im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
# else:
#     im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)

# cv2.namedWindow('Filtered Particles')
# cv2.namedWindow('Original Image')

# # Sliders for block size and C
# cv2.createTrackbar('Block Size', 'Filtered Particles', 7, 13, lambda x: None)
# cv2.createTrackbar('C', 'Filtered Particles', 20, 40, lambda x: None)

# while True:
#     block_slider = cv2.getTrackbarPos('Block Size', 'Filtered Particles')
#     c_slider = cv2.getTrackbarPos('C', 'Filtered Particles')

#     block_size = 2 * block_slider + 3
#     C = c_slider - 20

#     adaptive_thresh = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
#     )

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

#     labelarray, num_features = ndimage.label(cleaned)
#     sizes = ndimage.sum(cleaned, labelarray, range(1, num_features + 1))

#     min_area = 20
#     mask = np.zeros_like(cleaned)

#     for i, size in enumerate(sizes):
#         if size > min_area:
#             mask[labelarray == i + 1] = 255

#     label_filtered, final_particle_count = ndimage.label(mask)

#     display = im.copy()
#     centroids = ndimage.center_of_mass(mask, label_filtered, range(1, final_particle_count + 1))

#     for center in centroids:
#         if not np.isnan(center[0]) and not np.isnan(center[1]):
#             cv2.circle(display, (int(center[1]), int(center[0])), 5, (0, 255, 0), 2)

#     text = f'Particles Detected: {final_particle_count}'
#     cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     cv2.imshow('Filtered Particles', display)
#     cv2.imshow('Original Image', gray)  # Show original grayscale

#     print(f'Particles: {final_particle_count}, Block Size: {block_size}, C: {C}', end='\r')

#     key = cv2.waitKey(100)
#     if key == 27:
#         break

# cv2.destroyAllWindows()






# import cv2
# import numpy as np
# from scipy import ndimage
# from PIL import Image

# # Load image with PIL for cropping
# image_path = '/Users/prashansapkota/Documents/7-1-24 LT 5 hr 90kx15.tif'
# im_pil = Image.open(image_path)

# # Crop the image (left, top, right, bottom)
# width, height = im_pil.size
# crop_area = (0, 0, width, height - 100)
# im_pil = im_pil.crop(crop_area)

# # Convert to OpenCV format (numpy array, BGR)
# im = np.array(im_pil)
# if len(im.shape) == 2:
#     im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
# else:
#     im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)

# cv2.namedWindow('Filtered Particles')
# cv2.namedWindow('Original Image')

# # Sliders for block size and C
# cv2.createTrackbar('Block Size', 'Filtered Particles', 7, 13, lambda x: None)
# cv2.createTrackbar('C', 'Filtered Particles', 20, 40, lambda x: None)

# while True:
#     block_slider = cv2.getTrackbarPos('Block Size', 'Filtered Particles')
#     c_slider = cv2.getTrackbarPos('C', 'Filtered Particles')

#     block_size = 2 * block_slider + 3
#     C = c_slider - 20

#     adaptive_thresh = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
#     )

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

#     labelarray, num_features = ndimage.label(cleaned)
#     sizes = ndimage.sum(cleaned, labelarray, range(1, num_features + 1))

#     min_area = 20
#     mask = np.zeros_like(cleaned)

#     for i, size in enumerate(sizes):
#         if size > min_area:
#             mask[labelarray == i + 1] = 255

#     label_filtered, final_particle_count = ndimage.label(mask)

#     display = im.copy()
#     centroids = ndimage.center_of_mass(mask, label_filtered, range(1, final_particle_count + 1))

#     for center in centroids:
#         if not np.isnan(center[0]) and not np.isnan(center[1]):
#             cv2.circle(display, (int(center[1]), int(center[0])), 5, (0, 255, 0), 2)

#     text = f'Particles Detected: {final_particle_count}'
#     cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     cv2.imshow('Filtered Particles', display)
#     cv2.imshow('Original Image', gray)  # Show original grayscale

#     print(f'Particles: {final_particle_count}, Block Size: {block_size}, C: {C}', end='\r')

#     key = cv2.waitKey(100)
#     if key == 27:
#         break

# cv2.destroyAllWindows()





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

