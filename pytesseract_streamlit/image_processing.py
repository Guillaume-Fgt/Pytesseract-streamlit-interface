from io import BytesIO
import os
import cv2
from cv2 import Mat
import pytesseract
import numpy as np
import re
from config import tesseract_exec_path


# path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = tesseract_exec_path


class ImageProcessing:
    """methods to prepare an image using Opencv and extract its text using PyTesseract"""

    def __init__(self, uploaded_file: BytesIO) -> None:
        self.img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    def threshold_img(self, lower: int = 100, upper: int = 255) -> Mat:
        """change the colors of an image between two values."""
        lower = np.array([lower, lower, lower])
        upper = np.array([upper, upper, upper])
        thresh = cv2.inRange(self.img, lower, upper)
        return thresh

    def mask_img(self, image, struct_elem, choice_morph):
        """take an image and return a masked version (background deleted)"""
        struct_elem = getattr(cv2, struct_elem)
        choice_morph = getattr(cv2, choice_morph)
        kernel = cv2.getStructuringElement(struct_elem, (20, 20))
        morph = cv2.morphologyEx(image, choice_morph, kernel)
        masked = cv2.bitwise_and(self.img, self.img, mask=morph)
        return masked

    def adaptive_thresh(
        self, image: Mat, adaptiveMethod, thresholdType, blocksize: int, constant: int
    ):
        """turn an image in B&W and increase its contrast"""
        adaptiveMethod = getattr(cv2, adaptiveMethod)
        thresholdType = getattr(cv2, thresholdType)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adaptiv_threshold = cv2.adaptiveThreshold(
            gray, 255, adaptiveMethod, thresholdType, blocksize, constant
        )
        return adaptiv_threshold

    def dilate(self, image: Mat, iterations: int, gauss_blur: int, size: int):
        """take an image and transform it in B&W areas that will be
        used to delimitate rectangles (Region of Interest)
        """
        blur = cv2.GaussianBlur(image, (gauss_blur, gauss_blur), 0)
        threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[
            1
        ]
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        dilate = cv2.dilate(threshed, kernal, iterations=iterations)
        return dilate

    def find_contours(
        self,
        image: Mat,
        width_min: int,
        height_min: int,
        width_max: int,
        height_max: int,
    ):
        """identify areas of an image and draw its borders.
        Return the coordinates of each shape
        """
        cnts, hierarchy = cv2.findContours(
            image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        rois = []
        rect_num = 1
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if width_min < w < width_max and height_min < h < height_max:
                rectangle = cv2.rectangle(
                    self.img, (x, y), (x + w, y + h), color=(36, 255, 12), thickness=4
                )
                rois.append([[x, y, w, h], rect_num])
                num = cv2.putText(
                    self.img,
                    str(rect_num),
                    org=(x + 10, y + h - 10),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=2,
                    color=(36, 255, 12),
                    thickness=4,
                )
                rect_num += 1
        return rectangle, rois

    def contour_to_text(self, rois, psm: str, language: str):
        """use Pytesseract to extract text from an image."""
        psm_re = re.compile(r"\d+ ")
        psm = psm_re.match(psm)
        config_psm = "--psm " + psm[0]
        if language == "English":
            lang = "eng"
        else:
            lang = "fra"
        text = ""
        for roi in rois:
            x, y, w, h = roi[0]
            roi_search = self.img[y : y + h, x : x + w]
            text_py = pytesseract.image_to_string(
                roi_search, lang=lang, config=config_psm
            )
            text += f"***ROI n°{roi[1]}***\n"
            text += text_py + "\n"
        return text

    def save_text_to_file(self, text: str) -> None:
        path = os.getcwd() + "/result"
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/savetext.txt", "w", encoding="utf_8") as f:
            f.write(text)

    def save_image_to_file(self, rois: Mat) -> None:
        path = os.getcwd() + "/result"
        if not os.path.exists(path):
            os.makedirs(path)
        for roi in rois:
            x, y, w, h = roi[0]
            roi_img = self.img[y : y + h, x : x + w]
            cv2.imwrite(f"{path}/{x}.jpg", roi_img)
