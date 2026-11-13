import os
import cv2
import numpy as np
from djitellopy import Tello
# --- PARÁMETROS ---
ANCHO, ALTO = 360, 240 # tamaño de procesamiento de la imagen
AREA_REFERENCIA = [6000, 10000] # rango de área del rostro (px2) con
distancia correcta
# Ganancias del controlador PD: [Kp, Kd]
PD_X = [0.4, 0.4] # giro (yaw)
PD_Y = [0.4, 0.4] # subir / bajar
PD_Z = [0.004, 0.004] # avanzar / retroceder
# Límites de velocidad (el rango del dron es -100 a 100)
LIM_YAW = 50
LIM_VERTICAL = 35
LIM_AVANCE = 30
BATERIA_MINIMA = 30 # % mínimo para permitir el despegue
CONVERTIR_RGB = False # ponlo en True si los colores se ven invertidos
# --- CARGA DEL CLASIFICADOR (ruta relativa al script) ---
DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
RUTA_XML = os.path.join(DIR_SCRIPT, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(RUTA_XML)
if face_cascade.empty():
    raise SystemExit(f"Error: no se pudo cargar el clasificador en
{RUTA_XML}")

def buscar_rostro(img):
    """Detecta rostros y devuelve la imagen dibujada y [[cx, cy], area]
    del rostro más grande (el más cercano). Sin rostro devuelve [[0, 0],
0]."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
    )
    centros, areas = [], []
    for (x, y, w, h) in rostros:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        centros.append([cx, cy])
        areas.append(w * h)

if areas:
    i = areas.index(max(areas))
    return img, [centros[i], areas[i]]
return img, [[0, 0], 0]

def rastrear_rostro(info, ancho, alto, p_error):
    """Calcula las velocidades (adelante/atrás, arriba/abajo, giro) con un
    controlador PD. Devuelve (velocidades, errores_actuales)."""
    (x, y), area = info
    # Sin rostro detectado: el dron se queda quieto
    if x == 0 and y == 0:
        return [0, 0, 0], (0, 0, 0)
    error_x = x - ancho // 2
    error_y = alto // 2 - y

    if area < AREA_REFERENCIA[0]:
        error_z = AREA_REFERENCIA[0] - area # rostro lejos: avanzar
    elif area > AREA_REFERENCIA[1]:
        error_z = AREA_REFERENCIA[1] - area # rostro cerca: retroceder
    else:
        error_z = 0

    p_x, p_y, p_z = p_error

    yaw = PD_X[0] * error_x + PD_X[1] * (error_x - p_x)
    vertical = PD_Y[0] * error_y + PD_Y[1] * (error_y - p_y)
    avance = PD_Z[0] * error_z + PD_Z[1] * (error_z - p_z)

    yaw = int(np.clip(yaw, -LIM_YAW, LIM_YAW))
    vertical = int(np.clip(vertical, -LIM_VERTICAL, LIM_VERTICAL))
    avance = int(np.clip(avance, -LIM_AVANCE, LIM_AVANCE))

    return [avance, vertical, yaw], (error_x, error_y, error_z)

def main():
    me = Tello()
    me.connect()
    bateria = me.get_battery()
    print("Nivel de batería:", bateria, "%")
    if bateria < BATERIA_MINIMA:
    print(f"Batería por debajo del {BATERIA_MINIMA} %. Carga el dron
antes de volar.")
    return

    me.streamoff()
    me.streamon()
    frame_read = me.get_frame_read()

    modo_seguimiento = False
    volando = False
    p_error = (0, 0, 0)
    try:
        while True:
            frame = frame_read.frame
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
                
            if CONVERTIR_RGB:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = cv2.resize(frame, (ANCHO, ALTO))
            img, info = buscar_rostro(img)

            if modo_seguimiento and volando:
                velocidades, p_error = rastrear_rostro(info, ANCHO, ALTO, p_error)

                # send_rc_control(izq/der, adelante/atrás, arriba/abajo, giro)

                me.send_rc_control(0, velocidades[0], velocidades[1],

velocidades[2])
            else:
                p_error = (0, 0, 0)
                me.send_rc_control(0, 0, 0, 0) # también mantiene viva la
conexión

            estado = "SEGUIMIENTO: ACTIVO" if modo_seguimiento else "SEGUIMIENTO: INACTIVO"

            color = (0, 255, 0) if modo_seguimiento else (0, 0, 255)
            cv2.putText(img, estado, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("RoboMaster TT - Tracking Facial", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('t') and not volando:
                print("Comando: despegar")
                me.takeoff()
                volando = True
            elif key == ord('l') and volando:
                print("Comando: aterrizar")
                 modo_seguimiento = False
                me.send_rc_control(0, 0, 0, 0)
                me.land()
                volando = False
            elif key == ord(' '):
                modo_seguimiento = not modo_seguimiento
                print("Modo seguimiento:", modo_seguimiento)
            elif key == ord('e'):
                print("EMERGENCIA: motores cortados")
                me.emergency()
                volando = False
                modo_seguimiento = False
            elif key == ord('q'):
                print("Finalizando programa")
            break

    finally:
         # Se ejecuta siempre, incluso si el programa falla a mitad de vuelo
        try:
            me.send_rc_control(0, 0, 0, 0)
            if volando:
                me.land()
        me.streamoff()
        me.end()
    except Exception as e:
        print("Aviso al cerrar:", e)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()