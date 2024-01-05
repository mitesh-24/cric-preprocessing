import cv2
import numpy as np
# cap = cv2.VideoCapture(r"E:\college stuff\Final year project\CricShot10 dataset\cover\cover_0001.avi")
cap = cv2.VideoCapture(r"E:\college stuff\Final year project\CricShot10 dataset\cover\cover_0001.avi")

saliency = cv2.saliency.StaticSaliencyFineGrained_create()
# frame = cv2.imread(r"C:\Users\mites\Downloads\image.jpg")
while True:
    ret, frame = cap.read()
    (success, saliencyMap) = saliency.computeSaliency(frame)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    t,threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("Image", frame)
    cv2.imshow("Output", saliencyMap)
    cv2.imshow("Thresh", threshMap)
    if cv2.waitKey(1) == ord("q"):
        break
    # break

cv2.destroyAllWindows()