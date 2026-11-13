import cv2
import time
import os
import urllib.request
from djitellopy import Tello

# 1. Comprobar y descargar el archivo de rostros si no existe
xml_file = 'haarcascade_frontalface_default.xml'
if not os.path.exists(xml_file):
    print("Descargando archivo de detección de rostros...")
    url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    urllib.request.urlretrieve(url, xml_file)
    print("Descarga completada.")

# 2. Cargar el detector asegurando que el archivo local funcione
face_cascade = cv2.CascadeClassifier(xml_file)

if face_cascade.empty():
    print("Error crítico: El archivo XML está dañado o no se cargó correctamente.")
    exit()

# Crear la instancia del dron
dron = Tello()

try:
    print("Conectando con el Tello...")
    dron.connect()
    print(f"Batería: {dron.get_battery()}%")

    dron.streamon()
    frame_read = dron.get_frame_read()
    time.sleep(2)

    print("Iniciando cámara. Presiona 'q' en la ventana de video para salir...")

    while True:
        frame = frame_read.frame
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (720, 480))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Rostro", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Deteccion de Rostros - Tello", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Ocurrió un error en ejecución: {e}")

finally:
    dron.streamoff()
    cv2.destroyAllWindows()
    dron.end()
    print("Conexión finalizada.")