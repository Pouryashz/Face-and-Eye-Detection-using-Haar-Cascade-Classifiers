import cv2

# Load the input image
input_image = cv2.imread("Male.jpg")
# Image is read.

# Load the Haar cascade classifiers for face and eye detection
haar_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
haar_eye = cv2.CascadeClassifier("haarcascade_eye.xml")
# Haar cascade classifier face and eyes model are imported.

# Convert the input image to grayscale
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
# haar detection method only works on gray scale images.

# Detect faces in the grayscale image
faces = haar_face.detectMultiScale(gray_image, minNeighbors=8, scaleFactor=1.3)
# Haar face detection method is applied.

# Loop through each detected face
for x, y, w, h in faces:
    # Draw a rectangle around the detected face
    cv2.rectangle(input_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # A rectangle will be drawn around the detected face

    # Detect eyes within the face region
    eyes = haar_eye.detectMultiScale(gray_image)
    for ex, ey, ew, eh in eyes:
        # Draw a rectangle around each detected eye
        cv2.rectangle(input_image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# Display the final image with detected faces and eyes
cv2.imshow("Frame", input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
