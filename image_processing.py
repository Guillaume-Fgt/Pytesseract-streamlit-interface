import os
import re
from io import BytesIO

import cv2
import numpy as np
import pytesseract
from config import tesseract_exec_path
from cv2 import Mat
from cv2.typing import MatLike

# path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = tesseract_exec_path


class ImageProcessing:
    """methods to prepare an image using Opencv and extract its text using PyTesseract"""

    def load_image(self, uploaded_file: BytesIO) -> None:
        self.img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    def threshold_img(self, lower: int = 100, upper: int = 255) -> MatLike:
        """change the colors of an image between two values."""
        lower_b = np.array([lower, lower, lower])
        upper_b = np.array([upper, upper, upper])
        return cv2.inRange(self.img, lower_b, upper_b)

    def mask_img(self, image, struct_elem, choice_morph) -> MatLike:
        """take an image and return a masked version (background deleted)"""
        struct_elem = getattr(cv2, struct_elem)
        choice_morph = getattr(cv2, choice_morph)
        kernel = cv2.getStructuringElement(struct_elem, (20, 20))
        morph = cv2.morphologyEx(image, choice_morph, kernel)
        return cv2.bitwise_and(self.img, self.img, mask=morph)

    def adaptive_thresh(
        self,
        image: MatLike,
        adaptiveMethod,
        thresholdType,
        blocksize: int,
        constant: int,
    ) -> MatLike:
        """turn an image in B&W and increase its contrast"""
        adaptivemethod = getattr(cv2, adaptiveMethod)
        thresholdtype = getattr(cv2, thresholdType)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            gray, 255, adaptivemethod, thresholdtype, blocksize, constant
        )

    def dilate(
        self, image: MatLike, iterations: int, gauss_blur: int, size: int
    ) -> MatLike:
        """take an image and transform it in B&W areas that will be
        used to delimitate rectangles (Region of Interest)
        """
        blur = cv2.GaussianBlur(image, (gauss_blur, gauss_blur), 0)
        threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[
            1
        ]
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        return cv2.dilate(threshed, kernal, iterations=iterations)

    def find_contours(
        self,
        image: MatLike,
        width_min: int,
        height_min: int,
        width_max: int,
        height_max: int,
    ) -> tuple[Mat, Mat]:
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

    def contour_to_text(self, rois, psm: str, language: str) -> str:
        """use Pytesseract to extract text from an image."""
        psm_re = re.compile(r"\d+ ")
        psm_num = num if (num := psm_re.match(psm)) else ""
        config_psm = "--psm " + psm_num[0]
        lang = "eng" if language == "English" else "fra"
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
