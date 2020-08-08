from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import streamlit as st
from playsound import playsound
from PIL import Image


def detect_and_predict_mask(frame, faceNet, maskNet):

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > confidence_value:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:

        preds = maskNet.predict(faces)

    return (locs, preds)


image = Image.open('mask.png')
st.image(image, use_column_width=True, format='PNG')

html_temp = """
    <div style="background-color:#010200;padding:6px; margin-bottom: 15px;">
    <h2 style="color:white;text-align:center;">Mask Detection Application</h2>
    </div>
    """    
st.markdown(html_temp, unsafe_allow_html=True)


prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(
    ["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


maskNet = load_model("facemask_detector.model")

cuda = st.selectbox('NVIDIA CUDA GPU should be used?', ('True', 'False'))

USE_GPU = bool(cuda)

if USE_GPU:
    st.info("[INFO] setting preferable backend and target to CUDA...")
    faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

confidence_value = st.slider('Confidence:', 0.0, 1.0, 0.5, 0.1)
st.info("[INFO] loading face detector model...")
st.info("[INFO] loading face mask detector model...")

if st.button('Start'):

    image_placeholder = st.empty()

    st.success("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=700)

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):

            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            if mask > withoutMask:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)
                playsound('mask.mp3')

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        image_placeholder.image(
            frame, caption='Live Mask Detection Running..!', channels="BGR")
