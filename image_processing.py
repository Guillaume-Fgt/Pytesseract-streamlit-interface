import re
from io import BytesIO
from pathlib import Path

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

    def mask_img(self, image: MatLike, struct_elem: str, choice_morph: str) -> MatLike:
        """take an image and return a masked version (background deleted)"""
        struct_elem_int = getattr(cv2, struct_elem)
        choice_morph_int = getattr(cv2, choice_morph)
        kernel = cv2.getStructuringElement(struct_elem_int, (20, 20))
        morph = cv2.morphologyEx(image, choice_morph_int, kernel)
        return cv2.bitwise_and(self.img, self.img, mask=morph)

    def adaptive_thresh(
        self,
        image: MatLike,
        adaptivemethod: str,
        thresholdtype: str,
        blocksize: int,
        constant: int,
    ) -> MatLike:
        """turn an image in B&W and increase its contrast"""
        adaptivemethod_int = getattr(cv2, adaptivemethod)
        thresholdtype_int = getattr(cv2, thresholdtype)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            gray, 255, adaptivemethod_int, thresholdtype_int, blocksize, constant
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
    ) -> tuple[MatLike, list[list[int]]]:
        """identify areas of an image and draw its borders.
        Return the coordinates of each shape
        """
        cnts, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        rect = self.img
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if width_min < w < width_max and height_min < h < height_max:
                cv2.rectangle(
                    rect, (x, y), (x + w, y + h), color=(36, 255, 12), thickness=4
                )
                rois.append([x, y, w, h])
        return rect, rois

    def contour_to_text(self, rois: list[list[int]], psm: str, language: str) -> str:
        """use Pytesseract to extract text from an image."""
        psm_re = re.compile(r"\d+ ")
        psm_num = num if (num := psm_re.match(psm)) else ""
        config_psm = "--psm " + psm_num[0]
        lang = "eng" if language == "English" else "fra"
        text = ""
        for roi in enumerate(rois, start=1):
            rect_num, (x, y, w, h) = roi
            roi_search = self.img[y : y + h, x : x + w]
            text_py = pytesseract.image_to_string(
                roi_search, lang=lang, config=config_psm
            )
            text += f"***ROI n°{rect_num}***\n"
            text += text_py + "\n"
        return text

    def save_text_to_file(self, text: str) -> None:
        path = Path.cwd() / "result"
        if not path.exists():
            path.mkdir()
        path.joinpath("saved_text.txt").write_text(text)

    def save_image_to_file(self, rois: Mat) -> None:
        path = Path.cwd() / "result/ROIs"
        if not path.exists():
            path.mkdir()
        for roi in enumerate(rois, start=1):
            rect_num, (x, y, w, h) = roi
            roi_img = self.img[y : y + h, x : x + w]
            cv2.imwrite(f"{path}/{rect_num}.jpg", roi_img)
