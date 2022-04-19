import streamlit as st
from main import Image_Processing


def main():

    # st.set_page_config(layout="wide")
    st.header("PyTesseract Image Processing")
    st.subheader("Optical character recognition (OCR)")
    uploaded_file = st.sidebar.file_uploader("Choose a file")

    if uploaded_file:
        st.sidebar.subheader("Original image")
        st.sidebar.image(uploaded_file)
        image = Image_Processing(uploaded_file)

        col1, col2 = st.columns(2)
        with col2:
            st.subheader("Thresh settings")
            lower = st.slider("Lower bound", 0, 255, 100)
            upper = st.slider("Upper bound", 0, 255, 255)
        with col1:
            st.subheader("Thresh")
            thresh = image.threshold_img(lower, upper)
            st.image(thresh)

        col3, col4 = st.columns(2)
        with col4:
            st.subheader("Mask settings")
            choice_struct = st.radio(
                "Structuring element",
                ("MORPH_RECT", "MORPH_CROSS", "MORPH_ELLIPSE"),
                index=2,
            )
            choice_morph = st.radio(
                "Morphological operation",
                (
                    "MORPH_ERODE",
                    "MORPH_DILATE",
                    "MORPH_OPEN",
                    "MORPH_CLOSE",
                    "MORPH_GRADIENT",
                    "MORPH_TOPHAT",
                    "MORPH_BLACKHAT",
                    "MORPH_HITMISS",
                ),
                index=3,
            )
        with col3:
            st.subheader("Masked image")
            masked = image.mask_img(thresh, choice_struct, choice_morph)
            st.image(masked)

        col5, col6 = st.columns(2)
        with col6:
            st.subheader("Adaptive Threshold settings")
            choice_adapt_thresh = st.radio(
                "Adaptive Method",
                (
                    "ADAPTIVE_THRESH_MEAN_C",
                    "ADAPTIVE_THRESH_GAUSSIAN_C",
                ),
            )
            choice_thresh = st.radio(
                "Threshold type",
                (
                    "THRESH_BINARY",
                    "THRESH_BINARY_INV",
                ),
            )
            block = st.slider("Block size", 1, 99, 61, 2)
            constant = st.slider("Constant", 1, 100, 11)
        with col5:
            st.subheader("Adaptive Threshold")
            adapt_thresh = image.adaptive_thresh(
                masked, choice_adapt_thresh, choice_thresh, block, constant
            )
            st.image(adapt_thresh)

        col7, col8 = st.columns(2)
        with col8:
            st.subheader("Dilate settings")
            ite = st.slider("Number of iterations", 1, 4)
            gauss_blur = st.slider("Gaussian Blur", 1, 15, 1, 2)
            size = st.slider("Size of the structuring elements", 1, 40, 10)
        with col7:
            st.subheader("Dilated image")
            dilated = image.dilate(adapt_thresh, ite, gauss_blur, size)
            st.image(dilated)

        col9, col10 = st.columns(2)
        with col10:
            st.subheader("Contours settings")
            width_min = st.slider("Minimum width", 0, 3000, 100)
            width_max = st.slider("Maximum width", 0, 3000, 3000)
            height_min = st.slider("Minimum height", 0, 3000, 100)
            height_max = st.slider("Maximum height", 0, 3000, 3000)
        with col9:
            st.subheader("Contours detected")
            try:
                rectangles, rois = image.find_contours(
                    dilated, width_min, height_min, width_max, height_max
                )
                st.image(rectangles)
            except UnboundLocalError:
                st.write("No contour detected")
                text = ""

        col11, col12 = st.columns(2)
        with col12:
            st.subheader("PyTesseract settings")
            psm = st.radio(
                "Page segmentation modes",
                (
                    "0 - Orientation and script detection (OSD) only",
                    "1 - Automatic page segmentation with OSD",
                    "2 - Automatic page segmentation, but no OSD, or OCR",
                    "3 - Fully automatic page segmentation, but no OSD (Default)",
                    "4 - Assume a single column of text of variable sizes",
                    "5 - Assume a single uniform block of vertically aligned text",
                    "6 - Assume a single uniform block of text",
                    "7 - Treat the image as a single text line",
                    "8 - Treat the image as a single word",
                    "9 - Treat the image as a single word in a circle",
                    "10 - Treat the image as a single character",
                    "11 - Sparse text. Find as much text as possible in no particular order",
                    "12 - Sparse text with OSD",
                    "13 - Raw line. Treat the image as a single text line",
                ),
                index=3,
            )
            lang = st.radio("Language", ("English", "French"))
        with col11:
            with st.expander("Show extracted text"):
                text = image.contour_to_text(rois, psm, lang)
                if st.button("Save text to file"):
                    image.save_text_to_file(text)
                st.text(text)
            with st.expander("Show individual ROI"):
                if st.button("Save individual ROI"):
                    image.save_image_to_file(rois)
                for roi in rois:
                    x, y, w, h = roi[0]
                    roi_img = masked[y : y + h, x : x + w]
                    st.write(roi[1])
                    st.image(roi_img)