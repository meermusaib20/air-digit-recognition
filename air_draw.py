import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class DigitRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = DigitRecognizer()
model.load_state_dict(torch.load("digit_model.pth", map_location="cpu"))
model.eval()

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

canvas = np.zeros((480, 640), dtype=np.uint8)
prev_x, prev_y = None, None

print("Controls: p=predict | c=clear | q=quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        lm = result.hand_landmarks[0][8]
        x, y = int(lm.x * w), int(lm.y * h)

        if prev_x is not None:
            cv2.line(canvas, (prev_x, prev_y), (x, y), 255, 10)

        prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    combined = cv2.add(frame, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
    cv2.imshow("Air Digit Recognition", combined)

    key = cv2.waitKey(5) & 0xFF

    if key == ord('c'):
        canvas[:] = 0

    if key == ord('p'):
        img = cv2.resize(canvas, (28, 28))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, 1)

        print("Predicted Digit:", pred.item())

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
