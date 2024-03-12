from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

def nothing(x):
   pass 


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model(r"C:\Users\user\Documents\Student ID Logger\keras_model.h5", compile=False)

# Load the labels
class_names = open(r"C:\Users\user\Documents\Student ID Logger\labels.txt", "r").readlines()
target_class = "MSU ID"  # Class name for which to capture photo

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

low_threshold = 10
high_threshold = 50
# Create window for displaying the frame and sliders
cv2.namedWindow('Contour Detection with Trackbars')
# Minimum area for an ID card (adjust based on your ID card size)
min_id_area = 5000  # Adjust this value as needed
  # Create trackbars for low and high thresholds
cv2.createTrackbar('Low Threshold', 'Contour Detection with Trackbars', low_threshold, 500, nothing)
cv2.createTrackbar('High Threshold', 'Contour Detection with Trackbars', high_threshold, 500, nothing)
cv2.createTrackbar('Min ID Area', 'Contour Detection with Trackbars', min_id_area, 10000, nothing)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    low_threshold = cv2.getTrackbarPos('Low Threshold', 'Contour Detection with Trackbars')
    high_threshold = cv2.getTrackbarPos('High Threshold', 'Contour Detection with Trackbars')
    min_id_area = cv2.getTrackbarPos('Min ID Area', 'Contour Detection with Trackbars')
    # Convert frame to grayscale for Canny edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the biggest contour (assuming the ID card is the largest object)
    largest_contour = None
    max_area = 0

    for cnt in contours:
      area = cv2.contourArea(cnt)
      if area > max_area:
        max_area = area
        largest_contour = cnt

    # Check if a large enough contour is found
    if largest_contour is not None and max_area > min_id_area:
      # Approximate the contour with polygons
      approx = cv2.approxPolyDP(largest_contour, 0.08 * cv2.arcLength(largest_contour, True), True)

      # Check if the approximated contour has 4 corners (rectangle)
      if len(approx) == 4:
        # Check for convexity (ensure it's a reasonably shaped rectangle)
        if cv2.isContourConvex(approx):
          # Draw rectangle border around the ID card
          x, y, w, h = cv2.boundingRect(approx)
          cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)



    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    #print("Class:", class_name[2:], end="")
    #print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    # Capture photo if target class and confidence score is high

    
    if class_name == class_names[0]:
        # Generate unique filename with timestamp
        print("Im MSU ID")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
