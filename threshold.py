import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import io
from scipy import ndimage
from PIL import Image


st.set_page_config(page_title="Nano-Particle Detector", layout="wide")
st.title("Interactive Nano-Particle Detection")

# Load image with PIL for cropping
uploaded = st.file_uploader("Upload a microscopy image", type= ["tiff", "tif", "png", "jpg", "jpeg"])
if uploaded is not None: 
    im_pil = Image.open(uploaded)

    # Crop the image (left, top, right, bottom)
    width, height = im_pil.size
    CROP_PIXELS = 100
    im_pil_cropped = im_pil.crop((0, 0, width, height - CROP_PIXELS))

    # Parameters: Block Size and C for adaptive threshold
    block_slider = st.slider('Block Size', 0, 10, 0, help="Controls local thresholding area. Final value: 2*value+3 (odd numbers only).")
    c_slider = st.slider('Constant (C)', 0, 40, 0, help="Controls offset for adaptive threshold. Range: -20 to 20.")
    block_size = 2 * block_slider + 3
    C = c_slider - 20

    # Convert to OpenCV format (numpy array, BGR)
    im = np.array(im_pil_cropped)
    if len(im.shape) == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold and filtering
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

    # Particle Labeling and area filtering
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

    particle_centers = []
    for center in centroids:
        if not np.isnan(center[0]) and not np.isnan(center[1]):
            y, x = int(center[0]), int(center[1])
            cv2.circle(display, (x, y), 5, (0, 255, 0), 2)
            particle_centers.append((y,x))

    text = f'Particles Detected: {final_particle_count}'
    cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show results
    col1, col2 = st.columns(2)

    with col1:
        st.image(im_pil_cropped, caption="Original Image", use_container_width=True)

    with col2:
        st.image(display, channels="BGR", caption="Detected Image", use_container_width=True)
    st.write(text)

    # Download button for the annotated image
    display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    display_pil = Image.fromarray(display_rgb)
    buf = io.BytesIO()
    display_pil.save(buf, format='PNG')
    byte_im = buf.getvalue()
    st.download_button("Download Annotated Image", data=byte_im, file_name="particles_annotated.png", mime="image/png")

    # Download button for the particle annotation
    df = pd.DataFrame(particle_centers, columns=['y', 'x'])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="Download Particle Annotations (CSV)",
        data=csv_data,
        file_name="particle_annotations.csv",
        mime="text/csv"
    )

    # Calculate & plot histogram of pairwise centroid distances
    if len(particle_centers) >= 2:
        coords = np.array(particle_centers)
        dists = []
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                d = np.linalg.norm(coords[i] - coords[j])
                dists.append(d)
        fig, ax = plt.subplots()
        ax.hist(dists, bins=30)
        ax.set_title("Histogram of Particle-Particle Distances")
        ax.set_xlabel("Pixel Distance")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("Not enough particles to compute distance histogram.")
else: 
    st.info("Please upload an image to get started.")
