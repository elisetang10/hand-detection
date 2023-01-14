import cv2
import mediapipe as mp
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_util
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6)
tipIds = [4, 8, 12, 16, 20]
def drawHandLandmarks(image, hand_landmarks):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(
                image, landmarks, mp_hands.HAND_CONNECTIONS)
def countFingers(image, hand_landmarks, handNo=0):
    if hand_landmarks:
        landmarks = hand_landmarks[handNo].landmark
        # print(landmarks)
        fingers = []
        for lm_index in tipIds:
            # Get Finger Tip and Bottom y Position Value
            finger_tip_y = landmarks[lm_index].y
            finger_bottom_y = landmarks[lm_index - 2].y
            thumb_tip_x = landmarks[lm_index].x
            thumb_bottom_x = landmarks[lm_index - 2].x
            # Check if ANY FINGER is OPEN or CLOSED
            if lm_index != 4:
                if finger_tip_y < finger_bottom_y:
                    fingers.append(1)
                    # print("FINGER with id ", lm_index, " is Open")
                if finger_tip_y > finger_bottom_y:
                    fingers.append(0)
                    # print("FINGER with id ", lm_index, " is Closed")
                # else:
                #     if thumb_tip_x > thumb_bottom_x:
                #         fingers.append(1)
                #         # print("THUMB is Open")
                #     if thumb_tip_x < thumb_bottom_x:
                #         fingers.append(0)
                #         # print("THUMB is Closed")
        # print(fingers)
        totalFingers = fingers.count(1)
        # Display Text
        text = f'Fingers: {totalFingers}'
        cv2.putText(image, text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)
    results = hands.process(image)
    hand_landmarks = results.multi_hand_landmarks
    drawHandLandmarks(image, hand_landmarks)
    countFingers(image, hand_landmarks)
    cv2.imshow("Media Controller", image)
    if cv2.waitKey(1) == 32:
        break
cv2.destroyAllWindows()