
import numpy as np
import cv2
import tensorflow.keras as tf # type: ignore
import os
import pytesseract
import re
import sqlite3
from tkinter import *
from PIL import Image, ImageTk

# pip install tensorflow

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

conn = sqlite3.connect("logging.db")

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

    cap2 = cv2.VideoCapture(0)
    cap2.set(3,640)
    cap2.set(4,480)

    # width & height of webcam video in pixels -> adjust to your size
    # adjust values if you see black bars on the sides of capture window
    frameWidth = 1280
    frameHeight = 720

    # set width and height in pixels
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    # enable auto gain
    cap.set(cv2.CAP_PROP_GAIN, 0)

 

    # keeps program running forever until ctrl+c or window is closed
    while True:
        
        # disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Create the array of the right shape to feed into the keras model.
        # We are inputting 1x 224x224 pixel RGB image.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        ret, frame = cap.read()

        # crop to square for use with TM model
        margin = int(((frameWidth-frameHeight)/2))
        square_frame = frame[0:frameHeight, margin:margin + frameHeight]
        # resize to 224x224 for use with TM model
        resized_img = cv2.resize(square_frame, (224, 224))
        # convert image color to go to model
        model_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                 cv2.CHAIN_APPROX_NONE)
        d = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 0:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
        # for each one of the classes
        for i in range(0, len(classes)):
            # scale prediction confidence to %
            confidence.append(int(predictions[0][i]*100))
            # append classes and confidences to text for label
            conf_label += classes[i] + ": " + str(confidence[i]) + "%; "
            print(conf_label)
            if confidence[0] >= conf_threshold:
            
                text = pytesseract.image_to_string(thresh1)
                id_pattern = r"\d{4}-\d{4}"
                matches = re.findall(id_pattern, text)

                if matches:
                    print(matches[0])
        
            if (i == (len(classes)-1)):
                conf_label = ""
    
        
        # original video feed implementation
        cv2.imshow("Capturing", thresh1)
        cv2.waitKey(10)
         
        
        

    # terminate process 1
    p1.terminate()


if __name__ == '__main__':
    main()
    
    
