# Face and Eye Detection using Haar Cascade Classifiers

This Python script uses OpenCV's Haar cascade classifiers to detect faces and eyes in an image.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`) library installed (`pip install opencv-python`)

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/yourusername/face-eye-detection.git
    ```

2. Navigate to the project directory:

    ```
    cd face-eye-detection
    ```

3. Place your input image (`Male.jpg` or `Female.jpg`) in the project directory.

4. Ensure that the Haar cascade classifier files (`haarcascade_frontalface_default.xml` and `haarcascade_eye.xml`) are present in the project directory.

## Usage

Run the Python script:


The script will display the input image with rectangles drawn around detected faces and eyes.

## Notes

- The script assumes that the input image is named `Male.jpg`. Please replace it with your desired input image.
- Ensure that the Haar cascade classifier files are up-to-date and located in the project directory.
- Adjust parameters (`minNeighbors`, `scaleFactor`, etc.) in the code for optimal detection performance depending on your input images.
