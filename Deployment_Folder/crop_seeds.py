import cv2
import os
import numpy as np

def process_dataset(base_input_dir, base_output_dir):
    """
    Iterates through subfolders in input_dir (e.g., 'healthy', 'broken')
    and crops individual seeds into corresponding output subfolders.
    """
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    # Iterate through each class folder
    for class_name in os.listdir(base_input_dir):
        class_path = os.path.join(base_input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        output_class_path = os.path.join(base_output_dir, class_name)
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        print(f"Processing category: {class_name}")
        
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is None: continue

                # 1. Convert to grayscale and blur to reduce noise
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (7, 7), 0)
                
                # 2. Thresholding: Separate seeds from background
                # Using Otsu's method to automatically determine the best threshold
                # Note: Use cv2.THRESH_BINARY if seeds are dark on light background
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # 3. Clean up the image using morphological closing
                kernel = np.ones((5, 5), np.uint8)
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
                
                # 4. Find contours (the outlines of the seeds)
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                seed_count = 0
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    # Filter out small noise (adjust 1000 based on your image resolution)
                    if area > 1000: 
                        x, y, w, h = cv2.boundingRect(cnt)
                        
                        # 5. Crop with a small margin (padding)
                        margin = 15
                        crop = img[max(0, y-margin):y+h+margin, max(0, x-margin):x+w+margin]
                        
                        save_name = f"{os.path.splitext(filename)[0]}_seed_{seed_count}.jpg"
                        cv2.imwrite(os.path.join(output_class_path, save_name), crop)
                        seed_count += 1
                
                print(f"  - {filename}: Extracted {seed_count} seeds.")

if __name__ == "__main__":
    # Update these paths to match your local setup
    # 'raw_photos' should contain subfolders like 'healthy' and 'broken'
    process_dataset('raw_photos', 'dataset/training')
