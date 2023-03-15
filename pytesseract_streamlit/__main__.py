from streamlit.web import cli as stcli
from streamlit import runtime
import sys

from pytesseract_streamlit import image_processing
from pytesseract_streamlit import ui


def main():
    ui.ui(image_processing.ImageProcessing)


if __name__ == "__main__":
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0] + "/__main__.py"]
        sys.exit(stcli.main())
