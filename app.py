import cv2 as cv
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image

genai.configure(api_key="YOUR-API-KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

cap = cv.VideoCapture(0)
cap.set(4, 1280)
cap.set(3, 720)

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)


def getHandsInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        hand = hands[0] 
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        # print(fingers)
        return fingers, lmList
    else:
        return None


def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [0, 1, 1, 0, 0]:
        canvas = np.zeros_like(img)
    return current_pos, canvas


def sendToAi(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 1]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this maths problem.", pil_image])
        print(response.text)


prev_pos = None
canvas = None
img_combined = None
while True:
    success, img = cap.read()
    img = cv.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandsInfo(img)
    if info:
        fingers, lmList = info
        # print(fingers)
        prev_pos, canvas = draw(info, prev_pos, canvas)
        sendToAi(model, canvas, fingers)

    img_combined = cv.addWeighted(img, 0.7, canvas, 0.3, 0)

    cv.imshow("Image", img)
    cv.imshow("Canvas", canvas)
    cv.imshow("img_combined", img_combined)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break