import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
import logging
import json
import time
from datetime import datetime

# Configure logging
logging.basicConfig(filename='scanner.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

class BarcodeQRCodeScanner:
    def __init__(self, scan_duration=1, log_file='scanner.log', json_file='scanned_data.json'):
        self.scan_duration = scan_duration
        self.log_file = log_file
        self.json_file = json_file
        self.camera_index = 0  # Default camera index

    def decode(self, frame):
        """
        Decode barcodes and QR codes in the frame.
        
        Args:
            frame (numpy.ndarray): The frame from the video capture.
        
        Returns:
            list: A list of detected codes with type and data.
        """
        decoded_objects = pyzbar.decode(frame)
        detected_codes = []

        for obj in decoded_objects:
            points = obj.polygon
            if len(points) > 4:
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                points = hull.reshape((-1, 1, 2))
            points = np.array(points, dtype=np.int32)
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=3)
            
            code_type = obj.type
            code_data = obj.data.decode('utf-8')
            detected_codes.append({'type': code_type, 'data': code_data})
            
            # Display the detected code type and data on the frame
            text = f'{code_type}: {code_data}'
            cv2.putText(frame, text, (points[0][0][0], points[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return detected_codes

    def log_and_save(self, detected_codes):
        """
        Log the detected codes and save them to a JSON file.
        
        Args:
            detected_codes (list): A list of detected codes with type and data.
        """
        if detected_codes:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Log each code detected
            for code in detected_codes:
                code_type = code['type']
                code_data = code['data']
                logging.info(f"Detected {code_type}: {code_data}")
            
            # Save to JSON file
            with open(self.json_file, 'a') as f:
                for code in detected_codes:
                    code['timestamp'] = timestamp
                json.dump(detected_codes, f, indent=4)
                f.write('\n')  # Write a newline for each batch of detected codes

    def scan_codes(self):
        """
        Capture frames from the camera, detect barcodes and QR codes, and log their information.
        """
        cap = cv2.VideoCapture(self.camera_index)
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame from camera.")
                break
            
            detected_codes = self.decode(frame)
            
            # Log and save detected information
            self.log_and_save(detected_codes)
            
            # Display the frame
            cv2.imshow('Barcode & QR Code Scanner', frame)
            
            # Stop scanning if codes are detected or the scan duration has passed
            if detected_codes or (time.time() - start_time > self.scan_duration):
                break
            
            # Wait for the 'q' key to be pressed to exit manually
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Scanner stopped by user.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Scanning process completed.")

    def change_camera(self, camera_index):
        """
        Change the camera index.
        
        Args:
            camera_index (int): Index of the camera to be used.
        """
        self.camera_index = camera_index
        logging.info(f"Changed camera to index {camera_index}.")

    def set_scan_duration(self, duration):
        """
        Set the scan duration.
        
        Args:
            duration (int): Duration of the scan in seconds.
        """
        self.scan_duration = duration
        logging.info(f"Set scan duration to {duration} seconds.")

if __name__ == "__main__":
    # Example of usage
    scanner = BarcodeQRCodeScanner(scan_duration=5)
    scanner.scan_codes()

    # Additional features for changing camera and setting scan duration
    # Uncomment and use as needed
    scanner.change_camera(1)  # Change to camera with index 1
    scanner.set_scan_duration(3)  # Change scan duration to 3 seconds
