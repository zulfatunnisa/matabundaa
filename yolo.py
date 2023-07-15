from ultralytics import YOLO
import cv2
# import torch
import numpy as np
# from PIL import Image
# from torchvision.transforms import ToTensor
import wave

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO("model/best (1).pt")

threshold = 0.5
class_names = ['cutter','gunting','kompor','pisau']

# Inisialisasi webcam
cap = cv2.VideoCapture("http://172.20.10.13:8080/video")
#cap = cv2.VideoCapture("video/video_testing.mp4")

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    # Mengubah ukuran frame menjadi 50% dari ukuran asli
    resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    preds = model(resized_frame)[0]
    # detected_notajam = False

    # Menggambar kotak deteksi pada frame
    for pred in preds.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = pred

        if score > threshold:
            class_label = class_names[int(class_id)]
            cv2.rectangle(frame, (int(x1 * 2), int(y1 * 2)), (int(x2 * 2), int(y2 * 2)), (0, 0, 255), 4)
            cv2.putText(frame, class_label.upper(), (int(x1 * 2), int(y1 * 2 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

    # Menampilkan frame dengan deteksi
    cv2.imshow('Real-time Object Detection', frame)

    # Cek tombol keyboard 'q' untuk keluar
    if cv2.waitKey(1) == ord('q'):
        break

# Tutup webcam dan jendela tampilan
cap.release()
cv2.destroyAllWindows()