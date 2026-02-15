import cv2
import os
import numpy as np

def fix_images():
    """Convert all images in known_faces to 8-bit RGB JPG"""
    input_dir = "known_faces"
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_dir, filename)
            
            # Read image with alpha channel support
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                print(f"❌ Could not read {filename}, skipping...")
                continue
                
            # Remove alpha channel if exists
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
            # Convert 16-bit to 8-bit if needed
            if img.dtype == np.uint16:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
            # Convert to RGB and save as JPG
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            new_name = f"{os.path.splitext(filename)[0]}_fixed.jpg"
            cv2.imwrite(os.path.join(input_dir, new_name), rgb_img)
            print(f"✅ Converted {filename} to {new_name}")

if __name__ == "__main__":
    fix_images()
