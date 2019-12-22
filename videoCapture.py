import numpy as np
import cv2
import torch
from digitRecognizer import CNNModel
def get_frame_contour_thresh(frame, x, y, w, h):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(frame, (35, 35), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y + 60:y + 60 + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return frame, contours, thresh1


#load model
model = torch.load('digitRecognizer.pt')
model.eval()

# connect webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    original_frame = cv2.flip(frame, True)

    #draw small region used for prediction
    x, y, w, h = 0, 0, 350, 350
    cv2.rectangle(original_frame,(x, y+60), (x + w, y + 60 + h),(0,0,255),2)

    frame, contours, thresh = get_frame_contour_thresh(original_frame, x, y, w, h)
    prediction =''
    if(len(contours) > 0):
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 2000:
            #get the point of img have max contour
            x, y, w, h = cv2.boundingRect(contour)

            #pre-process data
            inputImg = thresh[y:y + h, x:x+ h]
            cv2.imshow('input img', inputImg)

            inputImg = cv2.resize(inputImg, (28, 28))
            inputImg = np.array(inputImg)
            # print(inputImg)
            inputImg = inputImg.flatten().reshape(1, 1, 28, 28) / 255 #normalize
            inputImg = torch.from_numpy(inputImg).type(torch.FloatTensor)

            #predicting
            out = model(inputImg)
            # print(torch.max(torch.nn.functional.softmax(out), 1)[0].data)
            if(torch.max(torch.nn.functional.softmax(out), 1)[0].data < 0.70):
                prediction = -1 #cannot predict
            else:
                prediction = torch.max(out, 1)[1].numpy()[0]


    cv2.putText(original_frame, "Prediction :  " + str(prediction), (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("original_frame",original_frame)
    cv2.imshow('thresh', thresh)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break