
from djitellopy import Tello

dron = Tello()

try:
    dron.connect()

    print("Batería:", dron.get_battery(), "%")
    print("Altura:", dron.get_height(), "cm")
    print("Temperatura:", dron.get_temperature(), "°C")

finally:
    dron.end()
