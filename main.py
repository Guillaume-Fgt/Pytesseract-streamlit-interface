from streamlit.web import cli as stcli
from streamlit import runtime
import sys

import image_processing
import ui


def main():
    ui.ui(image_processing.ImageProcessing)


if __name__ == "__main__":
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
