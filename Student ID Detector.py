import multiprocessing
import numpy as np
import cv2
import tensorflow.keras as tf
import pyttsx3
import os


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


# this process is purely for text-to-speech so it doesn't hang processor
def speak(speakQ, ):
    # initialize text-to-speech object
    engine = pyttsx3.init()
    # can adjust volume if you'd like
    volume = engine.getProperty('volume')
    engine.setProperty('volume', volume)  # add number here
    # initialize last_msg to be empty
    last_msg = ""
    # keeps program running forever until ctrl+c or window is closed
    while True:
        msg = speakQ.get()
        # clear out msg queue to get most recent msg
        while not speakQ.empty():
            msg = speakQ.get()
        # if most recent msg is different from previous msg
        # and if it's not "Background"
        if msg != last_msg and msg != "Background":
            last_msg = msg
            # text-to-speech say class name from labels.txt
            engine.say(msg)
            engine.runAndWait()
        if msg == "Background":
            last_msg = ""


def main():

    # read .txt file to get labels
    labels_path = f"{DIR_PATH}/Model/labels.txt"
    # open input file label.txt
    labelsfile = open(labels_path, 'r')

    # initialize classes and read in lines until there are no more
    classes = []
    line = labelsfile.readline()
    while line:
        # retrieve just class name and append to classes
        classes.append(line.split(' ', 1)[1].rstrip())
        line = labelsfile.readline()
    # close label file
    labelsfile.close()

    # load the teachable machine model
    model_path = f"{DIR_PATH}/Model/keras_model.h5"
    model = tf.models.load_model(model_path, compile=False)

    # initialize webcam video object
    cap = cv2.VideoCapture(0)

    # width & height of webcam video in pixels -> adjust to your size
    # adjust values if you see black bars on the sides of capture window
    frameWidth = 1280
    frameHeight = 720

    # set width and height in pixels
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    # enable auto gain
    cap.set(cv2.CAP_PROP_GAIN, 0)

    # creating a queue to share data to speech process
    speakQ = multiprocessing.Queue()

    # creating speech process to not hang processor
    p1 = multiprocessing.Process(target=speak, args=(speakQ, ), daemon="True")

    # starting process 1 - speech
    p1.start()

    # keeps program running forever until ctrl+c or window is closed
    while True:

        # disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Create the array of the right shape to feed into the keras model.
        # We are inputting 1x 224x224 pixel RGB image.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and help contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detection to find edges in the image
        edges = cv2.Canny(blurred, 50, 100)

        # Find contours in the edged image
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over the contours
        for contour in contours:
            # Approximate the shape
            epsilon = 0.08 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the contour has four corners and a specific aspect ratio (vertical rectangle)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 0.8 <= aspect_ratio <= 1.2:  # Adjust the aspect ratio range as needed
                    # Draw the contour on the original frame
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                    break  # Break the loop after finding the first suitable rectangle


        # Draw the contour on the original frame
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        # crop to square for use with TM model
        margin = int(((frameWidth-frameHeight)/2))
        square_frame = frame[0:frameHeight, margin:margin + frameHeight]
        # resize to 224x224 for use with TM model
        resized_img = cv2.resize(square_frame, (224, 224))
        # convert image color to go to model
        model_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        # turn the image into a numpy array
        image_array = np.asarray(model_img)
        # normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # load the image into the array
        data[0] = normalized_image_array

        # run the prediction
        predictions = model.predict(data)

        # confidence threshold is 90%.
        conf_threshold = 90
        confidence = []
        conf_label = ""
        threshold_class = ""
        
        # for each one of the classes
        for i in range(0, len(classes)):
            # scale prediction confidence to %
            confidence.append(int(predictions[0][i]*100))
            # append classes and confidences to text for label
            conf_label += classes[i] + ": " + str(confidence[i]) + "%; "
            print(conf_label)
            # prints last line
            if (i == (len(classes)-1)):
                conf_label = ""
            # if above confidence threshold, send to queue
            if confidence[i] > conf_threshold:
                speakQ.put(classes[i])
                threshold_class = classes[i]
                


        # original video feed implementation
        cv2.imshow("Capturing", frame)
        cv2.waitKey(10)

    # terminate process 1
    p1.terminate()


if __name__ == '__main__':
    main()
