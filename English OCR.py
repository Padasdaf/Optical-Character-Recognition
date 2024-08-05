import cv2
import pytesseract
from PIL import Image
import numpy as np
import os
import difflib
import string

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Update this path according to your installation

# Load a set of valid English words from a dictionary file or list
with open('/usr/share/dict/words', 'r') as file:
    valid_words = set(word.strip().lower() for word in file)

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

def get_closest_word(word, valid_words):
    # Find the closest match for the word in the valid_words set
    closest_matches = difflib.get_close_matches(word.lower(), valid_words, n=1, cutoff=0.7)
    return closest_matches[0] if closest_matches else word

def extract_text_from_image(image_path, language='eng'):
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Convert processed image to PIL Image for pytesseract
    pil_image = Image.fromarray(processed_image)

    # Perform OCR with specified language option
    text = pytesseract.image_to_string(pil_image, lang=language)

    # Replace invalid words with the closest valid word, preserving punctuation
    words = text.split()
    corrected_words = []
    for word in words:
        # Separate the word from punctuation
        stripped_word = word.strip(string.punctuation)
        prefix = word[:len(word) - len(stripped_word)]
        suffix = word[len(stripped_word):]
        
        # Get the closest valid word and reconstruct the word with punctuation
        corrected_word = get_closest_word(stripped_word, valid_words)
        corrected_words.append(prefix + corrected_word + suffix)

    corrected_text = ' '.join(corrected_words)

    return corrected_text

def main():
    image_path = '/Users/danie/Downloads/IMG_0785.jpg'  # Update this path to your image file
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    text = extract_text_from_image(image_path, language='eng')  # Use 'eng' for English
    
    print("Extracted Text:")
    print(text)

if __name__ == "__main__":
    main()

