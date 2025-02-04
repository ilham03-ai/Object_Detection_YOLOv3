import cv2  
    filters=64,
    kernel_size=3,
    padding="same",
    name="conv2d/1")
from yolov5 import YOLOv5

# Carica il modello YOLOv5 pre-addestrato
model = YOLOv5("yolov5s.pt")  # Assicurati che yolov5s.pt sia nella tua cartella

# Inizializza la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Esegui il rilevamento degli oggetti
    results = model.predict(frame)  # Usa model(frame), non model.predict(frame)

    # Visualizza il risultato con le etichette e le bounding boxes
    results.render()  # Disegna le bounding boxes sulle immagini

    # Mostra il frame con rilevamenti
    cv2.imshow("Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
