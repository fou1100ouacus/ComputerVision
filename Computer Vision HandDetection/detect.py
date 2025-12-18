
import cv2
import mediapipe as mp

# === 1 Open camera  ==============================================================
cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("ERROR: No camera found! Check connection or close Zoom/Teams.")
    exit()

# === 2 Mediapipe setup =========================================================
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mpDraw = mp.solutions.drawing_utils  #// green line hand draw 

tipIds = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky     ارقام نفط رووس الاصابع

# ===  3: Main loop ==================================================================
while True:
    success, img = cap.read() # read frame from camera
    if not success:
        print("Failed to read camera")
        break

    img = cv2.flip(img, 1)  # Mirror (feels natural) 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #to proccess image
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        lmList = [] # ليسته تحتوي علي احداثيات النقاط 20 لليد
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) 


#استخراج احداثيات كل نقطه ======================================================
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape 
                cx, cy = int(lm.x * w), int(lm.y * h)  #calulate every positon of land mark of fingers in x and y axies
                lmList.append([id, cx, cy])

            # Draw green circle on index finger tip
            cv2.circle(img, (lmList[8][1], lmList[8][2]), 15, (0, 255, 0), -1)



##==================================================================================================================
            # === Count fingers ===
            if len(lmList) == 21: #  نتاكد ان كل النقاط موجوده
                fingers = []

                # Thumb (uses X coordinate)
                if lmList[tipIds[0]][1] < lmList[tipIds[0]-2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other fingers (use Y coordinate)
                for i in range(1, 5):
                    if lmList[tipIds[i]][2] < lmList[tipIds[i]-2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total = fingers.count(1)

                # Show big number
                cv2.putText(img, str(total), (100, 150), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 15)

    # Show the camera window
    cv2.imshow('aya - Hand Counter (Press ESC to exit)', img)

    # Press ESC key to close
    if cv2.waitKey(1) == 27:
        break

# === Clean up ===
cap.release()
cv2.destroyAllWindows()
print("Thank you for using aya's app")










