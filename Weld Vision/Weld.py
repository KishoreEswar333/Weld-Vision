import os
import cv2
import numpy as np

# Define input and output directories
input_dir = r'D:\Projects\Auto measuring using Python\Weld defects images'
output_dir = 'detected_images'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define lower and upper bounds for brown color in HSV space
lower_brown = np.array([10, 50, 50])
upper_brown = np.array([30, 255, 255])

# Step 1: Image Acquisition
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.jfif', '.webp')):
        # Read image from input directory
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        # Step 2: Image Preprocessing for dark spots
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 3: Thresholding to detect dark spots
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

        # Step 4: Finding contours for dark spots
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 5: Drawing bounding boxes around dark spots
        for contour in contours:
            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)

            # Draw bounding box (red color)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)  # BGR color for red is (0, 0, 255)

        # Convert image to HSV color space for detecting brown areas
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only brown colors
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

        # Bitwise-AND mask_brown and original image
        brown_areas = cv2.bitwise_and(image, image, mask=mask_brown)

        # Convert brown_areas to grayscale for contour detection
        gray_brown = cv2.cvtColor(brown_areas, cv2.COLOR_BGR2GRAY)

        # Step 6: Finding contours for brown areas
        contours_brown, _ = cv2.findContours(gray_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 7: Drawing bounding boxes around brown areas
        for contour in contours_brown:
            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)

            # Draw bounding box (brown color)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 1)  # BGR color for brown is (0, 165, 255)

        # Step 8: Save the image with highlighted defects in the output directory
        output_filename = f'defects_{os.path.splitext(filename)[0]}.png'  # Change file extension to .png
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, image)

        print(f'Defects detected in {filename}. Saved as {output_path}')

print("Defect detection and saving complete.")
