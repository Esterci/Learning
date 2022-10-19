from serial import Serial

serialPort = Serial(port = "/dev/ttyUSB0", baudrate=9800)

print(serialPort.readline())


