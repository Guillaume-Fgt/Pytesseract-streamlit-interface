# Pytesseract-streamlit-interface

Webapp to retrieve text from images using OpenCV and Pytesseract. The interface is made with Streamlit.

The steps to obtain text are as followed:
  - load an image using button on the side bar
  - Using OpenCV, the image is processed in order to define ROI: Region Of Interest. 
    This is the parts of the image that will be send to Pytesseract for text detection.
    They will appear in green with a number as overlay. Tweaking the settings will change their number and shape.
  - You can change Pytesseract page segmentation mode and language to possibly improve text detection relevance.
    On the left column, you will see text extracted with corresponding ROI number and on the right the cropped image of the ROI.
    You have a button on top of each column to save text and images as file. By default, it is saved in the "result" directory of the project.
    
![Animation](https://user-images.githubusercontent.com/66461774/165033937-b0ba5251-01e0-4a82-90bd-6898889aae4c.gif)

How to use it:

* clone this repo
* create a venv and activate it
* execute this command indide the folder of the pyproject file:
pip install .
* with the venv activated, run
* python pytesseract_streamlit