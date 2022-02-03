import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
device = torch.device('cuda')
path = r'/home/iamshri/PycharmProjects/intent/yolov5/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path)
model.to(device)
names = model.names
mp_hands = mp.solutions.hands  # Holistic model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic  # Drawing utilities


def draw_plots(frame, results):
    for res in results:  # plot bounding boxes and include labels
        if res['class'] == 3:  # Book
            l = int(res['xmin'])
            t = int(res['ymin'])
            r = int(res['xmax'])
            b = int(res['ymax'])
            text_in_image = res['name']
            cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 1)
            cv2.putText(frame, text_in_image, (l, t), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 120), 1,
                        cv2.LINE_AA)
            return l, t, r, b, res['name']

        elif res['class'] == 1:  # Mug
            l1 = int(res['xmin'])
            t1 = int(res['ymin'])
            r1 = int(res['xmax'])
            b1 = int(res['ymax'])
            text_in_image = res['name']
            cv2.rectangle(frame, (l1, t1), (r1, b1), (255, 0, 0), 1)
            cv2.putText(frame, text_in_image, (l1, t1), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 120), 1,
                        cv2.LINE_AA)

        elif res['class'] == 2:  # Mugs
            l2 = int(res['xmin'])
            t2 = int(res['ymin'])
            r2 = int(res['xmax'])
            b2 = int(res['ymax'])
            text_in_image = res['name']
            # cv2.rectangle(frame, (l2, t2), (r2, b2), (255, 0, 255), 1)
            # cv2.putText(frame, text_in_image, (l2, t2), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 120), 1,
            #             cv2.LINE_AA)

        elif res['class'] == 0:
            l3 = int(res['xmin'])
            t3 = int(res['ymin'])
            r3 = int(res['xmax'])
            b3 = int(res['ymax'])
            text_in_image = res['name']
            cv2.rectangle(frame, (l3, t3), (r3, b3), (0, 255, 0), 1)
            cv2.putText(frame, text_in_image, (l3, t3), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 120), 1,
                        cv2.LINE_AA)
        else:
            pass


def Detection_pipeline(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    coords = holistic.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, coords


def draw_landmarks(image, coords):
    mp_drawing.draw_landmarks(image, coords.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2,
                                                     circle_radius=2))  # Draw left hand connections
    mp_drawing.draw_landmarks(image, coords.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    # mp_drawing.draw_landmarks(image, coords.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, coords.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


def create_white_background(image):
    empty = np.zeros([255, 255, 3], dtype=np.uint8)
    filling = empty.fill(255)
    cv2.imshow('Plot background window', filling)


cap = cv2.VideoCapture(0)
i = 0
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=.5) as holistic:
    while True:
        _, img = cap.read()
        image = cv2.flip(img, 1)
        img_, coords = Detection_pipeline(image, holistic)
        image2fill = img_.shape
        image2fill = np.zeros(image2fill, dtype=np.uint8)
        image2fill.fill(255)
        draw_landmarks(image2fill, coords)
        draw_landmarks(img_, coords)
        detect_obj = model(img_[..., ::-1])
        results = detect_obj.pandas().xywhn[0].to_dict(orient='records')
        rect_bbox = detect_obj.pandas().xyxy[0].to_dict(orient='records')
        l, t, r, b, text_to_display = draw_plots(img_, rect_bbox)

        cv2.imshow('Output', img_)
        cv2.imwrite('image{}.jpeg'.format(i), image2fill)
        cv2.rectangle(image2fill, (l, t), (r, b), (0, 0, 255), 1)
        cv2.putText(image2fill, text_to_display, (l, t), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 120), 1, cv2.LINE_AA)
        cv2.imshow('Plot Media', image2fill)
        cv2.imwrite('image{}.jpeg'.format(i), img_)
        cv2.imwrite('plot{}.jpeg'.format(i), image2fill)
        i += 1

        if cv2.waitKey(1) != ord('q'):
            continue
        cap.release()
        cv2.destroyAllWindows()
