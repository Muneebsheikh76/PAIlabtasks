import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def to_rgb(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def gaussian_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def rotate(image, angle=30):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, matrix, (w, h))
def translate(image, tx=100, ty=70):
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, matrix, (w, h))

def shear(image, shearX=-0.15, shearY=0):
    h, w = image.shape[:2]
    matrix = np.float32([[1, shearX, 0], [0, 1, shearY]])
    return cv2.warpAffine(image, matrix, (w, h))

def normalize(image):
    b, g, r = cv2.split(image)
    b_norm = cv2.normalize(b.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
    g_norm = cv2.normalize(g.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
    r_norm = cv2.normalize(r.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
    return cv2.merge((b_norm, g_norm, r_norm))

def edge_detection(image):
    image_rgb = to_rgb(image)
    return cv2.Canny(image_rgb, 100, 700)

def log_transform(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    c = 255 / (np.log(1 + np.max(img_gray)))
    log_transformed = c * np.log(1 + img_gray)
    return np.array(log_transformed, dtype=np.uint8)

def gamma_correction(image, gamma=1.2):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gamma_corrected = np.array(255 * (img_gray / 255) ** gamma, dtype='uint8')
    return gamma_corrected

def contrast_stretch(image):
    def pixelVal(pix, r1, s1, r2, s2):
        if (0 <= pix and pix <= r1):
            return (s1 / r1) * pix
        elif (r1 < pix and pix <= r2):
            return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
        else:
            return ((255 - s2) / (255 - r2)) * (pix - r2) + s2
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    r1, s1, r2, s2 = 70, 0, 140, 255
    pixelVal_vec = np.vectorize(pixelVal)
    return pixelVal_vec(img_gray, r1, s1, r2, s2).astype(np.uint8)

def hist_equalization(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(img_gray)

st.title('OpenCV Image Processing App')

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png', 'webp'])

functions = [
    'Gaussian Blur',
    'Grayscale',
    'Rotate',
    'Translate',
    'Shear',
    'Normalize',
    'Edge Detection',
    'Log Transform',
    'Gamma Correction',
    'Contrast Stretching',
    'Histogram Equalization'
]

selected_function = st.selectbox('Select a function to perform', functions)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(to_rgb(image), caption='Original Image', use_column_width=True)

    if selected_function == 'Gaussian Blur':
        result = gaussian_blur(image)
        st.image(to_rgb(result), caption='Gaussian Blurred', use_column_width=True)
    elif selected_function == 'Grayscale':
        result = grayscale(image)
        st.image(result, caption='Grayscale', use_column_width=True, channels='GRAY')
    elif selected_function == 'Rotate':
        result = rotate(image)
        st.image(to_rgb(result), caption='Rotated', use_column_width=True)
    elif selected_function == 'Translate':
        result = translate(image)
        st.image(to_rgb(result), caption='Translated', use_column_width=True)
    elif selected_function == 'Shear':
        result = shear(image)
        st.image(to_rgb(result), caption='Sheared', use_column_width=True)
    elif selected_function == 'Normalize':
        result = normalize(image)
        st.image(result, caption='Normalized', use_column_width=True)
    elif selected_function == 'Edge Detection':
        result = edge_detection(image)
        st.image(result, caption='Edges', use_column_width=True, channels='GRAY')
    elif selected_function == 'Log Transform':
        result = log_transform(image)
        st.image(result, caption='Log Transformed', use_column_width=True, channels='GRAY')
    elif selected_function == 'Gamma Correction':
        gamma = st.slider('Gamma', 0.1, 3.0, 1.2, 0.1)
        result = gamma_correction(image, gamma)
        st.image(result, caption=f'Gamma Corrected (gamma={gamma})', use_column_width=True, channels='GRAY')
    elif selected_function == 'Contrast Stretching':
        result = contrast_stretch(image)
        st.image(result, caption='Contrast Stretched', use_column_width=True, channels='GRAY')
    elif selected_function == 'Histogram Equalization':
        result = hist_equalization(image)
        st.image(result, caption='Histogram Equalized', use_column_width=True, channels='GRAY')
