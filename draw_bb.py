import cv2
import os
import argparse

# Global states
mode = 'CROP'  # Modes: 'CROP', 'ANNOTATE'
drawing_bbox = False
crop_x, crop_y = 0, 0
ix, iy = -1, -1
bbox = []

img_full = None
img_display = None
img_crop = None

def mouse_callback(event, x, y, flags, param):
    global mode, crop_x, crop_y, drawing_bbox, ix, iy, bbox, img_display, img_crop, img_full

    h_full, w_full = img_full.shape[:2]

    if mode == 'CROP':
        if event == cv2.EVENT_MOUSEMOVE:
            # Clamp the 640x640 window so it doesn't go out of bounds
            crop_x = max(0, min(x - 320, w_full - 640))
            crop_y = max(0, min(y - 320, h_full - 640))
            
            img_display = img_full.copy()
            cv2.rectangle(img_display, (crop_x, crop_y), (crop_x + 640, crop_y + 640), (0, 255, 255), 2)

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Lock in the crop and switch modes
            img_crop = img_full[crop_y:crop_y+640, crop_x:crop_x+640]
            img_display = img_crop.copy()
            mode = 'ANNOTATE'
            print("Crop locked. Now draw your bounding box.")

    elif mode == 'ANNOTATE':
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_bbox = True
            ix, iy = x, y
            bbox = []

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing_bbox:
                img_display = img_crop.copy()
                cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing_bbox = False
            cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
            xmin, ymin = min(ix, x), min(iy, y)
            xmax, ymax = max(ix, x), max(iy, y)
            bbox = [xmin, ymin, xmax, ymax]

def main(image_filename, class_id=0):
    global img_full, img_display, img_crop, bbox, mode
    
    os.makedirs('images', exist_ok=True)
    os.makedirs('labels', exist_ok=True)
    
    # Check if file exists in current directory
    if not os.path.exists(image_filename):
        print(f"Error: {image_filename} not found in the current folder.")
        return

    base_name = os.path.splitext(image_filename)[0]
    img_full = cv2.imread(image_filename)
    img_display = img_full.copy()
    
    # Use WINDOW_NORMAL so it fits on your screen if 1080p is too big
    cv2.namedWindow('Annotator', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Annotator', mouse_callback)
    
    print("--- STEP 1: CROP ---")
    print("Move mouse to position the 640x640 window.")
    print("Left Click : Lock crop region.")
    print("\n--- STEP 2: ANNOTATE ---")
    print("Click & Drag : Draw Box.")
    print("'s'          : Save outputs & Exit.")
    print("'r'          : Reset to Step 1 (Crop).")
    print("'q'          : Quit.")
    
    while True:
        cv2.imshow('Annotator', img_display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') and bbox and mode == 'ANNOTATE':
            xmin, ymin, xmax, ymax = bbox
            x_center = ((xmin + xmax) / 2.0) / 640.0
            y_center = ((ymin + ymax) / 2.0) / 640.0
            w = (xmax - xmin) / 640.0
            h = (ymax - ymin) / 640.0
            
            img_save_path = f"images/{base_name}_crop.png"
            txt_save_path = f"labels/{base_name}_crop.txt"
            
            cv2.imwrite(img_save_path, img_crop)
            with open(txt_save_path, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                
            print(f"Saved -> {img_save_path}")
            print(f"Saved -> {txt_save_path}")
            break
            
        elif key == ord('r'):
            print("Resetting to crop mode...")
            mode = 'CROP'
            bbox = []
            img_display = img_full.copy()
            
        elif key == ord('q'):
            print("Quit.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()

    print("Opening file picker...")
    # Open a file dialog to select the image
    file_path = filedialog.askopenfilename(
        title="Select your 1080p Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if file_path:
        print(f"Loading {file_path}...")
        main(file_path, class_id=0)
    else:
        print("You didn't select a file. Quitting.")