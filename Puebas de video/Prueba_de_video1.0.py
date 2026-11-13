import cv2
import time
from djitellopy import Tello

dron = Tello()

try:
    print("Conectando con el Tello...")
    dron.connect()
    print(f"Batería: {dron.get_battery()}%")

    # 1. Encender el stream de video
    dron.streamon()

    # 2. Obtener el capturador de fotogramas
    frame_read = dron.get_frame_read()

    print("Mostrando video en vivo. Presiona la tecla 'q' en la ventana para salir...")

    # Bucle continuo para refrescar la ventana frame a frame
    while True:
        # Capturar el frame actual
        frame = frame_read.frame
        # Corregir el color azulado (Convertir BGR a RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Redimensionar la imagen para ajustar el tamaño en pantalla (opcional)
        frame = cv2.resize(frame, (720, 480))

        # Mostrar el frame en la ventana "Camara Tello"
        cv2.imshow("Camara Tello", frame)

        # Si se presiona la tecla 'q', romper el bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cerrando transmisión de video...")
            break

except Exception as e:
    print(f"Error durante la ejecución: {e}")

finally:
    # Apagar el stream de video, cerrar ventanas de OpenCV y finalizar conexión
    dron.streamoff()
    cv2.destroyAllWindows()
    dron.end()
    print("Conexión y video finalizados.")