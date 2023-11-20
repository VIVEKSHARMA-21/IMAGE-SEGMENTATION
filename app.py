import cv2
import numpy as np
import streamlit as st
import copy

def hex_to_bgr(hex_color):
    color = []
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("Invalid HEX color format. It should be in the format RRGGBB.")


    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    color.append(b)
    color.append(g)
    color.append(r)

    return color  # BGR format

st.set_page_config(page_title="Image Segmentation BY VIVEK")

st.title("Image Segmentation")

selectbox = st.sidebar.selectbox(
    "Which operation do you want to perform?",
    ("Grabcut Algorithm", "Color Masking", "Otsu Thresholding", "Morphological Operations")
)


if selectbox == "Color Masking":
  st.subheader(selectbox)
  image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"], key="cm")

  if image_file is not None:

    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    col1, col2 = st.columns(2)
    with col1:
      lcolor = st.color_picker('Pick A Lower Color', '#5A0000')
    with col2:
        ucolor = st.color_picker('Pick A Upper Color', '#FFFAFA')

    lcolor = hex_to_bgr(lcolor)
    ucolor = hex_to_bgr(ucolor)
    lower_color = np.array(lcolor)
    upper_color = np.array(ucolor)

    color_mask = cv2.inRange(image, lower_color, upper_color)

    segmented_image = cv2.bitwise_and(image, image, mask=color_mask)

    col1, col2 = st.columns(2)
    with col1:
      st.image(image, channels="BGR", caption="Original Image")

    with col2:
      st.image(segmented_image, channels="BGR", caption="Result Image")

if selectbox == "Otsu Thresholding":
  st.subheader(selectbox)
  image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"], key="ot")

  if image_file is not None:

    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, segmented_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    col1, col2 = st.columns(2)
    with col1:
      st.image(image, channels="BGR", caption="Original Image")

    with col2:
      st.image(segmented_image, caption="Result Image")

if selectbox == "Morphological Operations":
  st.subheader(selectbox)
  image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"], key="mo")

  if image_file is not None:

    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8) 
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2) 

    bg = cv2.dilate(closing, kernel, iterations = 1) 

    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0) 
    ret, fg = cv2.threshold(dist_transform, 0.02 * dist_transform.max(), 255, 0) 

    col1, col2 = st.columns(2)
    with col1:
      st.image(image, channels="BGR", caption="Original Image")

    with col2:
      st.image(fg, clamp=True, caption="Result Image")

if selectbox == "Grabcut Algorithm":
    st.subheader(selectbox)
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], key="gc")

    if image_file is not None:

        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        imagecopy = image.copy()

        col1, col2 = st.columns(2)

        ix = col1.number_input("X - Co-ordinate", 0, step=10, key="ix")
        iy = col1.number_input("Y - Co-ordinate", 0, step=10, key="iy")
        w = col1.number_input("Width", 0, step=10, key="w")
        h = col1.number_input("Height", 0, step=10, key="h")

        color = (0, 0, 255)

        cv2.rectangle(imagecopy, (ix, iy), (ix + w, iy + h), color, thickness=2)
        col2.image(imagecopy, channels="BGR", caption="Select Area of Interest")
        st.write("")

        if ix > 0 and iy > 0 and w > 0 and h > 0:

            mask = np.zeros(image.shape[:2], np.uint8)
            backgroundModel = np.zeros((1, 65), np.float64)
            foregroundModel = np.zeros((1, 65), np.float64)
            rectangle = (ix, iy, w, h)

            cv2.grabCut(image, mask, rectangle, backgroundModel, foregroundModel, 3, cv2.GC_INIT_WITH_RECT)

            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            segmented_image = image * mask2[:, :, np.newaxis]

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, channels="BGR", caption="Original Image")

            with col2:
                st.image(segmented_image, channels="BGR", caption="Result Image")