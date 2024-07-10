import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector


st.page_link("./main.py",label="Go to Back",)

st.title("Hand Detection")
run = st.checkbox("Run")
FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(0)

detector = HandDetector(detectionCon=0.8)

while True:
    s,img = cam.read()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    hands,img = detector.findHands(img)

    FRAME_WINDOW.image(img)

    if run == False:
        cam.release()
        break
    
