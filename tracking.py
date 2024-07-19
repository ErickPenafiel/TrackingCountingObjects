import cv2
from ultralytics import YOLO
from sort import Sort
import numpy as np
import time
import torch

from utils import *

if __name__ == '__main__':
    cap = cv2.VideoCapture("/videos/test3.mp4")
    print(torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    model = YOLO('yolov8m-seg.pt').to(device)
    line = [(100, 500), (1050, 500)]

    tracker = Sort()
    trackerMalezas = Sort()
    start_time = time.time()
    frame_count = 0
    count_id = []
    count_obj = {}
    count_obj_intersection = 0

    while True:
        ret, frame = cap.read()

        resultados = model(frame)
        
        result = resultados[0].plot()        

        for res in resultados:
            clases = res.boxes.cls.cpu().numpy().astype(int)
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            mascaras = res.masks.xy
            # print("MASCARAS: ", mascaras)
            # print("CLASES: ", clases)
            # print("BOXES: ", boxes)
            trackerMasks, intersection_counter = associate_boxes_with_tracks(boxes, tracker.update(boxes), clases, mascaras, line, count_id, count_obj)
            trackerMasks = np.array(trackerMasks, dtype=object)
            # print("TRACKER MASKS: ", trackerMasks)
            count_obj_intersection += intersection_counter            

            for xmin, ymin, xmax, ymax, track_id, clase, mascaras in trackerMasks:
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                track_id = int(track_id)

                cv2.putText(img=frame, text=f"Id:{track_id} Clase: {classes[clase]}", org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,255,0), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(img=frame, text=f"FPS: {fps:.2f} Intersection Count: {count_obj_intersection}", org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        for cls in count_obj:
            i = list(count_obj.keys()).index(cls)
            cv2.putText(img=frame, text=f"{classes[cls]}: {count_obj[cls]}", org=(10, 70 + i * 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        
        cv2.line(frame, line[0], line[1], (0, 255, 0), thickness=2)

        cv2.imshow("Deteccion y segmentacion", frame)
        
        frame_count += 1
        if(cv2.waitKey(1) == 27):
            break

cap.release()
cv2.destroyAllWindows()