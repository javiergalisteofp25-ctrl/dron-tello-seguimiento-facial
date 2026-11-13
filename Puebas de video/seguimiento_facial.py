import cv2
import time
import os
import numpy as np
from djitellopy import Tello

# 1. Cargar el detector de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Cargar tu foto de referencia para extraer tu patrón biométrico
RUTA_MI_FOTO = "mi_foto.jpg"

if not os.path.exists(RUTA_MI_FOTO):
    print(f"Error: No se encuentra la foto '{RUTA_MI_FOTO}' en la carpeta del proyecto.")
    print("Guarda una foto de tu cara con el nombre 'mi_foto.jpg' para continuar.")
    exit()

# Cargar imagen de referencia y convertir a escala de grises
img_referencia = cv2.imread(RUTA_MI_FOTO)
gray_referencia = cv2.cvtColor(img_referencia, cv2.COLOR_BGR2GRAY)
faces_ref = face_cascade.detectMultiScale(gray_referencia, 1.1, 5)

if len(faces_ref) == 0:
    print("Error: No se detectó ninguna cara en 'mi_foto.jpg'. Usa una foto más clara.")
    exit()

(rx, ry, rw, rh) = faces_ref[0]
rostro_referencia = cv2.resize(gray_referencia[ry:ry+rh, rx:rx+rw], (100, 100))

# Crear un reconocedor facial de OpenCV (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create() if hasattr(cv2, 'face') else None

# Si no está disponible el módulo extra de cv2, usamos comparación por plantilla estructurada (Histogram Matching)
def evaluar_similitud(rostro_detectado):
    rostro_norm = cv2.resize(rostro_detectado, (100, 100))
    # Calcular correlación de histograma de textura entre tu foto y el rostro actual
    hist_ref = cv2.calcHist([rostro_referencia], [0], None, [256], [0, 256])
    hist_det = cv2.calcHist([rostro_norm], [0], None, [256], [0, 256])
    cv2.normalize(hist_ref, hist_ref)
    cv2.normalize(hist_det, hist_det)
    similitud = cv2.compareHist(hist_ref, hist_det, cv2.HISTCMP_CORREL)
    return similitud > 0.75 # Umbral de tolerancia de coincidencia

# Inicializar dron Tello
dron = Tello()

try:
    print("Conectando con el Tello...")
    dron.connect()
    bateria = dron.get_battery()
    print(f"Batería: {bateria}%")

    if bateria < 35:
        print("Batería muy baja para volar con seguridad. Cancelando despegue.")
    else:
        dron.streamon()
        frame_read = dron.get_frame_read()
        time.sleep(2)

        print("Despegando en 3 segundos...")
        time.sleep(3)
        dron.takeoff()
        dron.move_up(40)

        # Crear ventana explícita de OpenCV en el escritorio
        cv2.namedWindow("Tello - Reconocimiento y Seguimiento Facial", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tello - Reconocimiento y Seguimiento Facial", 720, 480)

        print("Sistema ACTIVO. Mostrando video en pantalla. Presiona 'q' para salir/aterrizar.")

        centro_x = 720 // 2
        centro_y = 480 // 2
        area_minima = 6000  # Si es menor, el dron avanza hacia ti
        area_maxima = 9000  # Si es mayor, el dron retrocede

        while True:
            frame = frame_read.frame
            if frame is None:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (720, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

            v_izq_der = 0
            v_ade_atras = 0
            v_arr_abajo = 0
            v_giro = 0

            if len(faces) > 0:
                # Filtrar la cara más cercana (más grande)
                cara_principal = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = cara_principal
                
                # Extraer la región del rostro detectado para verificar identidad
                rostro_recorte = gray[y:y+h, x:x+w]
                es_mi_rostro = evaluar_similitud(rostro_recorte)

                cx = x + w // 2
                cy = y + h // 2
                area = w * h

                if es_mi_rostro:
                    # VERDE: Rostro verificado (TÚ)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, "USUARIO CONFIRMADO (SIGUIENDO)", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Cálculo de control de vuelo
                    error_x = cx - centro_x
                    v_giro = int(error_x * 0.15)

                    error_y = cy - centro_y
                    v_arr_abajo = int(error_y * -0.2)

                    if area < area_minima:
                        v_ade_atras = 20
                    elif area > area_maxima:
                        v_ade_atras = -20
                    else:
                        v_ade_atras = 0

                    v_giro = max(-35, min(35, v_giro))
                    v_arr_abajo = max(-35, min(35, v_arr_abajo))

                    # Enviar comandos de movimiento
                    dron.send_rc_control(v_izq_der, v_ade_atras, v_arr_abajo, v_giro)

                else:
                    # ROJO: Desconocido (Se queda quieto en el aire)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "DESCONOCIDO (IGNORANDO)", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    dron.send_rc_control(0, 0, 0, 0)
            else:
                # Si no ve a nadie, frena motores
                dron.send_rc_control(0, 0, 0, 0)
                cv2.putText(frame, "Buscando tu rostro...", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Mostrar pantalla de video en vivo en el ordenador
            cv2.imshow("Tello - Reconocimiento y Seguimiento Facial", frame)

            # Tecla de parada de emergencia
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Aterrizaje solicitado por el usuario.")
                break

except Exception as e:
    print(f"Error detectado durante la ejecución: {e}")

finally:
    print("Finalizando programa y aterrizando...")
    dron.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)
    dron.land()
    dron.streamoff()
    cv2.destroyAllWindows()
    dron.end()
    print("Conexión cerrada de forma segura.")