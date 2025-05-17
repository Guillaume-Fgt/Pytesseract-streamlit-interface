import sys

import image_processing
import ui
from streamlit import runtime
from streamlit.web import cli as stcli


def main() -> None:
    ui.ui(image_processing.ImageProcessing())


if __name__ == "__main__":
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
