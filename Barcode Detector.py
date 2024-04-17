#!/usr/bin/env python
# coding: utf-8


from pyzbar import pyzbar
import cv2

def tesseract_scan(image):
    captured_image = cv2.
    cap_scr = cv2.cvtColor(cap_scr, cv2.COLOR_RGB2BGR)
    return cap_scr


cap = cv2.VideoCapture(0)   # read video file or stream. 0 means capturing from built in webcam.

while cap.isOpened(): # successful capture
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is not None:        
        detectedBarcodes = readBarcodes(frame,barcodeNumbers)        
        cv2.imshow('barcodes', detectedBarcodes)
        
        if cv2.waitKey(1) & 0xFF == 27: # exit when ESC is pressed
            break
    else:
        print('unable to read next frame')
        break

print(barcodeNumbers)
cap.release()
cv2.destroyAllWindows()