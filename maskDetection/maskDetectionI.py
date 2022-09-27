# ========================================================================= #
#                          Project: Mask Detection                          #
# ========================================================================= #

# Import Dependencies
import cv2
import numpy as np

# Load the image and get its width and height
image = cv2.imread('images/test_image.jpg')
imageHeight, imageWidth = image.shape[:2]

# Turn our image into blob
imageBlob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB = True, crop = False)

# Define the labels
labels = ["good", "bad"]

# Define different colors for per label
colors = ['0, 255, 255', '0, 0, 255']

# Get per value in colors list
colors = [np.array(color.split(',')).astype('int') for color in colors]
colors = np.array(colors)

# Create the model
maskDetectionImageModel = cv2.dnn.readNetFromDarknet(r"C:\Users\gorke\Desktop\maskDetection\pretrainedModel\yolov3_mask.cfg",
                                                     r"C:\Users\gorke\Desktop\maskDetection\pretrainedModel\yolov3_mask_last.weights")

# Get the class names of our model
modelLayers = maskDetectionImageModel.getLayerNames()

# Get the output layer of our model
outputLayer = [modelLayers[layer - 1] for layer in maskDetectionImageModel.getUnconnectedOutLayers()]

# Set up the input layer
maskDetectionImageModel.setInput(imageBlob)

# Create the detection layers
detectionLayers = maskDetectionImageModel.forward(outputLayer)

# Create the necessary lists for Non-Maximum Suppression
ids = []
boxes = []
confidences = []

# Creating the detection algorithm
for detectionLayer in detectionLayers:
    for objectDetection in detectionLayer:
        scores = objectDetection[5:]
        predictedId = np.argmax(scores)
        confidence = scores[predictedId]

        if confidence > 0.20:
            label = labels[predictedId]
            boundingBox = objectDetection[0:4] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
            (boxCenterX, boxCenterY, boxWidth, boxHeight) = boundingBox.astype('int')

            startX = int(boxCenterX - (boxWidth / 2))
            startY = int(boxCenterY - (boxHeight / 2))

            ids.append(predictedId)
            confidences.append(float(confidence))
            boxes.append([startX, startX, int(boxWidth), int(boxHeight)])

# Non-Maximum Suppression
maxIds = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

for maxId in maxIds:
    maxClassId = maxId
    box = boxes[maxClassId]

    startX, startY, boxWidth, boxHeight = box[0], box[1], box[2], box[3]

    predictedId = ids[maxClassId]
    label = labels[predictedId]
    confidence = confidences[maxClassId]

    endX = startX + boxWidth
    endY = startY + boxHeight
    boxColor = colors[predictedId]
    boxColor = [int(each) for each in boxColor]

    label = '{}: {:.2f}%'.format(label, confidence * 100)
    print('Predicted Object {}'.format(label))
    cv2.rectangle(image, (startX, startY), (endX, endY), boxColor, 1)
    cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, boxColor, 1)

cv2.imshow('Detection Window', image)
cv2.waitKey(0)