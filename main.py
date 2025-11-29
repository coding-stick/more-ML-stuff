import cv2, math
import mediapipe as mp
from pyo import *
 

s = Server().boot()  # start audio server
s.start()

##### REMEMBER TO LOAD YOUR SONG.WAV !!! #####

sf = SfPlayer("song.wav", speed=1, loop=True, mul=0.5)
sf.out()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

pitch = 1.0
volume = 1.0
speed = 1.0

right_mid = None
left_mid = None

def to_screen(x, y, w, h):
    sx = int(x * w)
    sy = int(y * h)
    return (sx, sy)

def find_midpoint(p1,p2):
    mx = (p1[0] + p2[0]) // 2
    my = (p1[1] + p2[1]) // 2
    return (mx, my)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        success, frame = cap.read()
        h,w,_ = frame.shape

        if not success:
            break
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks
        results = hands.process(img_rgb)

        # Draw landmarks
        if results.multi_hand_landmarks and results.multi_handedness:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                # Get the matching handedness
                hand_label = results.multi_handedness[i].classification[0].label
                
                h, w, c = frame.shape
                
                
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                '''
                # Draw landmark points + index numbers
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.putText(frame, str(id), (cx+10, cy+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                '''

                # Now do distances
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]

                xA, yA = to_screen(thumb.x, thumb.y, w, h)
                xB, yB = to_screen(index.x, index.y, w, h)

                if hand_label == "Right":
                    if len(results.multi_hand_landmarks) == 1:
                        pitch = 1.0
                    volume = math.dist((xA, yA), (xB, yB))
                    right_mid = find_midpoint((xA, yA), (xB, yB))
                    cv2.putText(frame, f'Volume: {int(volume)}', (xA, yA), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    if len(results.multi_hand_landmarks) == 1:
                        volume = 1.0
                    pitch = math.dist((xA, yA), (xB, yB))
                    left_mid = find_midpoint((xA, yA), (xB, yB))
                    cv2.putText(frame, f'Pitch: {int(pitch)}', (xA, yA), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if right_mid and left_mid and len(results.multi_hand_landmarks) == 2:
                    speed = math.dist(right_mid, left_mid)     
                    cv2.putText(frame, f'Speed: {int(speed)}', find_midpoint(right_mid, left_mid), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.line(frame, right_mid, left_mid, (0, 255, 0), 2)
                else:
                    speed = 1.0
                cv2.line(frame, (xA, yA), (xB, yB), (255, 0, 0), 2)
                print(pitch, volume, speed)
                


        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # SCALING & UPDATE SOUND
        if volume!=1.0:
            sf.mul = volume / 300
        else:
            sf.mul = 1.0
        if speed!=1.0:
            sf.speed = speed / 100  
        else:
            sf.speed = 1.0
        if pitch!=1.0:
            harm = Harmonizer(sf, transpo=(pitch/5 - 20)).out()
            print(pitch/5-20)

cap.release()
cv2.destroyAllWindows()
s.stop()