import numpy as np

page_segmentation_modes = np.array(
    [
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
    ]
)

morphological_operation = np.array(
    [
        "MORPH_ERODE",
        "MORPH_DILATE",
        "MORPH_OPEN",
        "MORPH_CLOSE",
        "MORPH_GRADIENT",
        "MORPH_TOPHAT",
        "MORPH_BLACKHAT",
        "MORPH_HITMISS",
    ]
)

structuring_element = np.array(["MORPH_RECT", "MORPH_CROSS", "MORPH_ELLIPSE"])

adaptive_method = np.array(["ADAPTIVE_THRESH_MEAN_C", "ADAPTIVE_THRESH_GAUSSIAN_C"])

threshold_type = np.array(
    [
        "THRESH_BINARY",
        "THRESH_BINARY_INV",
    ]
)

tesseract_exec_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
