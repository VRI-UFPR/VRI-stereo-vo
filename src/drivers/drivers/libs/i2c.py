import board

# SCL, SDA
BUSES = [
    {"sda": board.SDA_1, "scl": board.SCL_1, "default": True,},
    {"sda": board.SDA, "scl": board.SCL, "default": True,},
] 

def I2C(bus_id=1, frequency=100000):
    """
    I2C factory function to return an I2C object based on the bus number.
    """

    bus = BUSES[bus_id]

    if bus["default"]:
        from .busio import I2C
        return I2C(scl=bus["scl"], sda=bus["sda"], frequency=frequency)
    else:
        from .adafruit_bitbangio import I2C
        return I2C(scl=bus["scl"], sda=bus["sda"], frequency=frequency)