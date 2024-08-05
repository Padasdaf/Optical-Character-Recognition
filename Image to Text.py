import cv2
import pytesseract
from PIL import Image
import numpy as np
import os

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Update this path according to your installation

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Apply dilation and erosion to remove noise
    kernel = np.ones((1, 1), np.uint8)
    img_dilated = cv2.dilate(thresh, kernel, iterations=1)
    img_eroded = cv2.erode(img_dilated, kernel, iterations=1)

    return img_eroded

def extract_text_from_image(image_path, language='chi_sim'):
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Convert processed image to PIL Image for pytesseract
    pil_image = Image.fromarray(processed_image)

    # Perform OCR with specified language option
    text = pytesseract.image_to_string(pil_image, lang=language)

    return text

def main():
    image_path = '/Users/danie/Downloads/chinese-about.png'  # Update this path to your image file
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    text = extract_text_from_image(image_path, language='chi_sim')  # Use 'chi_sim' for Simplified Chinese
    
    print("Extracted Text:")
    print(text)

if __name__ == "__main__":
    main()
