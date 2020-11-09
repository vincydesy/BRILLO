import cv2
import argparse
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from Create_Openface_Model import *
import math

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=True,
                help='percorso al file di configurazione di yolo (yolov3.cfg)')
ap.add_argument('-w', '--weights', required=True,
                help='percorso per i pesi pre-allenati yolo (yolov3.weights)')
ap.add_argument('-cl', '--classes', required=True,
                help='percorso del file di testo contenente i nomi delle classi (yolov3.txt)')
ap.add_argument('-ca', '--cascade', required=True,
                help='modello per rilevare i volti (haarcascade_frontalface_default.xml)')
ap.add_argument('-wo', '--weightsopen', required=True,
                help='percorso per i pesi pre-allenati di openface (openface_weights.h5)')
args = vars(ap.parse_args())
np.set_printoptions(threshold=np.inf)


# Utilità

def face_recognition():
    
    def preprocess_image(image_path):
        img = load_img(image_path, target_size=(96, 96))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def get_output_layers(net):
        layer_names = net.getLayerNames()
        outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return outputlayers

    def draw_bounding_box(frame, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        confidence = confidences[i]
        color = colors[class_ids[i]]
        cv2.rectangle(frame, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (200, 200, 200), 2)

    # Load Yolo
    net = cv2.dnn.readNet(args["weights"],args["config"])
    classes = []
    with open(args["classes"],"r") as f:
        classes = [line.strip() for line in f.readlines()]

    colors= np.random.uniform(0,255,size=(len(classes),3))

    # loading image and font
    webcam = cv2.VideoCapture(0)  # 0 for 1st webcam, "input.mp4" for registered video
    font = cv2.FONT_HERSHEY_PLAIN

    while (webcam.isOpened()):
        status, frame = webcam.read()

        if (not status):
            print("Error reading frame")
            exit()

        height, width = frame.shape[:2]
        # detecting objects

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224,224)), 0.00392, (224,224))
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                id=str(classes[class_id])
                if (confidence > 0.5) and (id=='person'):
                    # object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])  # put all rectangle areas
                    confidences.append(float(confidence))  # how confidence was that object detected and show that percentage
                    class_ids.append(class_id)  # name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                draw_bounding_box(frame,class_ids[i],confidence,round(x),round(y),round(x + w),round(y + h))

        cv2.imshow("Video", frame)
        cv2.imwrite("Output.jpg",frame) # print last frame
        key = cv2.waitKey(1)  # wait 1ms the loop will start again and we will process the next frame

        if key==27:  # esc key stops the process
            break;

    webcam.release()
    cv2.destroyAllWindows()

    # Crop image to bounding box first and then to the face
    if (not boxes or len(boxes)!=1):
        return 0,0;
    else:
        img = cv2.imread('Output.jpg')
        first_crop = img[round(y):round(y+h), round(x):round(x+w)]
        cv2.imwrite("Output.jpg", first_crop)

        # Load the cascade
        face_cascade = cv2.CascadeClassifier(args["cascade"])
        img = cv2.imread('Output.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            second_crop = img[y:y + h, x:x + w]
        cv2.imwrite('Output.jpg', second_crop)

        #load Openface and image

        openface_model = create_openface_model()
        openface_model.load_weights(args["weightsopen"])
        outputyolo = 'Output.jpg'

        #preprocessing image with one shot training and print array

        img_vector = openface_model.predict(preprocess_image(outputyolo))[0, :]
        matrix = np.load("Array.npy")
        img_vector = math.sqrt(1 / 1) * matrix * img_vector
        return 1, img_vector