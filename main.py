import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/Users/prashansapkota/Documents/Nano-particle-segmentation/threshold/input_images/7-1-24 LT 3 hr 90kx17.tif')  # Replace with your filename
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(gray.ravel(), bins=256, range=(0, 256), color='gray')
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
