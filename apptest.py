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

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

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
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    
    if class_name == class_names[0]:
        # Generate unique filename with timestamp
        print("MSU ID DETECTED")
        pass
    elif class_name == class_names[1]:
        print("OTHER ID DETECTED")
        pass
    else:
        print("NO ID DETECTED")
        pass

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
