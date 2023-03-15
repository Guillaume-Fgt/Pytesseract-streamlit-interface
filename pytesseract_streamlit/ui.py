from typing import Protocol
import streamlit as st
from pytesseract_streamlit import config
from cv2 import Mat


class Processing(Protocol):
    def threshold_img(self, lower: int, upper: int):
        ...

    def mask_img(self, image, struct_elem, choice_morph):
        ...

    def adaptive_thresh(
        self, image: Mat, adaptiveMethod, thresholdType, blocksize: int, constant: int
    ):
        ...

    def dilate(self, image: Mat, iterations: int, gauss_blur: int, size: int):
        ...

    def find_contours(
        self,
        image: Mat,
        width_min: int,
        height_min: int,
        width_max: int,
        height_max: int,
    ):
        ...

    def contour_to_text(self, rois, psm: str, language: str):
        ...

    def save_text_to_file(self, text: str) -> None:
        ...

    def save_image_to_file(self, rois: Mat) -> None:
        ...


def ui(Processing) -> None:
    """define the steamlit UI"""
    # st.set_page_config(layout="wide")
    st.header("PyTesseract Image Processing")
    st.subheader("Optical character recognition (OCR)")
    uploaded_file = st.sidebar.file_uploader("Choose an image file")

    if uploaded_file:
        st.sidebar.subheader("Original image")
        st.sidebar.image(uploaded_file)
        image = Processing(uploaded_file)

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
                config.structuring_element,
                index=2,
            )
            choice_morph = st.radio(
                "Morphological operation",
                config.morphological_operation,
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
                config.adaptive_method,
            )
            choice_thresh = st.radio(
                "Threshold type",
                config.threshold_type,
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

        col11, col12, col13 = st.columns(3)
        with col13:
            st.subheader("PyTesseract settings")
            psm = st.radio(
                "Page segmentation modes",
                config.page_segmentation_modes,
                index=3,
            )
            lang = st.radio("Language", ("English", "French"))
        with col11:
            with st.expander("Show extracted text"):
                text = image.contour_to_text(rois, psm, lang)
                if st.button("Save text to file"):
                    image.save_text_to_file(text)
                st.text(text)
        with col12:
            with st.expander("Show individual ROI"):
                if st.button("Save individual ROI"):
                    image.save_image_to_file(rois)
                for roi in rois:
                    x, y, w, h = roi[0]
                    roi_img = masked[y : y + h, x : x + w]
                    st.write(roi[1])
                    st.image(roi_img)
