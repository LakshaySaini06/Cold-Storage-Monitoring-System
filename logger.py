import serial
import csv
import os
from datetime import datetime

PORT = 'COM9'
BAUD = 115200
OUTPUT_FILE = 'sensor_data.csv'

ser = serial.Serial(PORT, BAUD)
print("Collecting data...")

file_exists = os.path.isfile(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0

with open(OUTPUT_FILE, 'a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(['timestamp', 'temperature', 'humidity'])

    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            values = line.split(',')
            if len(values) == 3:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                writer.writerow([timestamp, values[1], values[2]]) 
                file.flush()
                print(f"{timestamp} | Temp: {values[1]}°C | Humidity: {values[2]}%")
        except KeyboardInterrupt:
            print("Stopped.")
            break
        except Exception as e:
            print("Error:", e)