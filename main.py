import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time


cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

# Importing all images
imgBackground = cv2.resize(cv2.imread("Resources/Background.png"), (1920, 1080))
imgGameOver = cv2.resize(cv2.imread("Resources/gameOver.png"), (1920, 1080))
imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.resize(cv2.imread("Resources/bat1.png", cv2.IMREAD_UNCHANGED), (50, 250))
imgBat2 = cv2.resize(cv2.imread("Resources/bat2.png", cv2.IMREAD_UNCHANGED), (50, 250))

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
ballPos = [100, 100]
speedX = 25
speedY = 25
gameOver = False
score = [0, 0]

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1920, 1080))  # Resize img to match the size of imgBackground
    imgRaw = img.copy()

    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # with draw

    # Overlaying the background image
    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

    # Check for hands
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 580)  # Adjusted for 1920x1080 resolution

            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, imgBat1, (150, y1))  # Adjusted position
                if 150 < ballPos[0] < 150 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 30
                    score[0] += 1

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, imgBat2, (1770 - w1, y1))  # Adjusted position
                if 1770 - 2*w1 < ballPos[0] < 1770 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] -= 30
                    score[1] += 1

    # Game Over
    if ballPos[0] < 40 or ballPos[0] > 1860:  # Adjusted for 1920x1080 resolution
        gameOver = True

    if gameOver:
        img = imgGameOver
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (915, 520), cv2.FONT_HERSHEY_COMPLEX,
                    2.5, (255, 0, 0), 5)  # Adjusted position slightly upwards and color to blue


    # If game not over move the ball
    else:

        # Move the Ball
        if ballPos[1] >= 780 or ballPos[1] <= 10:  # Adjusted for 1920x1080 resolution
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # Draw the ball
        img = cvzone.overlayPNG(img, imgBall, ballPos)

        cv2.putText(img, str(score[0]), (600, 980),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)  # Adjusted position
        cv2.putText(img, str(score[1]), (1320, 980),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)  # Adjusted position

    img[850:1050, 20:380] = cv2.resize(imgRaw, (360, 200))  # Adjusted position and size



    cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballPos = [100, 100]
        speedX = 25
        speedY = 25
        gameOver = False
        score = [0, 0]
        imgGameOver = cv2.resize(cv2.imread("Resources/gameOver.png"), (1920, 1080))  # Adjusted size
        time.sleep(2)  # 2-second delay

