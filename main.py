import cv2
import os

alg = "haarcascade_frontalface_default.xml"
name = cv2.CascadeClassifier(alg)

test_image = "image.png"
img = cv2.imread(test_image, 0)

# Ensure the directory exists
if not os.path.exists("stored-faces"):
    os.makedirs("stored-faces")

faces = name.detectMultiScale(
    img, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
)

j = 0

for x, y, w, h in faces:
    cropped_image = img[y:y+h, x:x+w]
    target_file = "stored-faces/" + str(j) + '.jpg'
    cv2.imwrite(target_file, cropped_image)
    j += 1
