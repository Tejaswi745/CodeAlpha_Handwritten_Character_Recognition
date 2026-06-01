import cv2
import torch
import torch.nn as nn
import numpy as np

# ================= CNN MODEL =================
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 26)  # 26 alphabets

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(-1, 64 * 5 * 5)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# ================= LOAD MODEL =================
model = CNNModel()
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()

labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ================= LOAD IMAGE =================
img = cv2.imread("test.png")

if img is None:
    print("❌ Image not found!")
    exit()

# convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blur (important)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# threshold (important)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# DEBUG VIEW
cv2.imshow("THRESHOLD", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ================= FIND LETTERS =================
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    print("❌ No letters detected!")
    exit()

# sort left → right
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

word = ""

# ================= PROCESS EACH LETTER =================
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # ignore very small noise
    if w < 5 or h < 5:
        continue

    letter = thresh[y:y+h, x:x+w]

    # make square (center)
    h1, w1 = letter.shape

    if h1 > w1:
        diff = h1 - w1
        pad = diff // 2
        letter = cv2.copyMakeBorder(letter, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    else:
        diff = w1 - h1
        pad = diff // 2
        letter = cv2.copyMakeBorder(letter, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

    # extra padding
    letter = cv2.copyMakeBorder(letter, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

    # resize to 28x28
    letter = cv2.resize(letter, (28, 28))

    # normalize
    letter = letter.astype("float32") / 255.0

    # to tensor
    tensor = torch.tensor(letter).unsqueeze(0).unsqueeze(0)

    # predict
    with torch.no_grad():
        output = model(tensor)
        _, pred = torch.max(output, 1)

    predicted_letter = labels[pred.item()]
    print("Detected:", predicted_letter)

    word += predicted_letter

# ================= FINAL OUTPUT =================
print("\n🔥 Predicted Word:", word)