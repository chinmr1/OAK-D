import os
import random
import glob
import cv2
import numpy as np
import glob
import albumentations as A
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# --- CONFIG ---
NUM_IMAGES_TO_GENERATE = 10000 # Adjust as needed, but be mindful of storage and generation time
IMG_SIZE = 640

SOURCE_CUBES_DIR = os.path.join('.', 'Alpha_Cube_Cropped') # This should contain your cropped cube PNGs with alpha channel
BG_DIR = os.path.join('.', 'Background')
bg_files = glob.glob(os.path.join(BG_DIR, '*.[jp][pn][g]')) # Grabs .jpg, .png, .jpeg
if not bg_files:
    raise ValueError(f"No images found in {BG_DIR}. Check your path.")

OUTPUT_IMG_DIR = os.path.join('.', 'Dataset', 'images', 'train')
OUTPUT_LBL_DIR = os.path.join('.', 'Dataset', 'labels', 'train')

DATASET_ROOT = os.path.join('.', 'Dataset')
for split in ['train', 'val']:
    os.makedirs(os.path.join(DATASET_ROOT, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, 'labels', split), exist_ok=True)

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

import albumentations as A
import random

bg_transform = A.Compose([
    # 1. ALWAYS do this so the math doesn't break
    A.RandomCrop(width=IMG_SIZE, height=IMG_SIZE), 
    
    # 2. Apply these 80% (4/5) of the time. 20% (1/5) of the time, it skips this entirely.
    A.Compose([
        A.RandomGamma(gamma_limit=(98, 102), p=0.1),
        A.ColorJitter(brightness= 0.1, contrast=0.01, saturation=0.05, hue=0.05, p=0.3)
    ], p=0.8) 
])

global_transform = A.Compose([
    # If you also want the global noise/blur to skip 1/5 of the time, nest it too:
    A.Compose([
        A.ISONoise(color_shift=(0.0, 0.02), intensity=(0.02, 0.2), p=0.3), 
        A.MotionBlur(blur_limit=2, p=0.2)
    ], p=0.8)
])

# Load all cube PNGs
cube_files = [os.path.join(SOURCE_CUBES_DIR, f) for f in os.listdir(SOURCE_CUBES_DIR) if f.endswith('.png')]
cubes = [Image.open(f).convert("RGBA") for f in cube_files]

def generate_background():
    bg_path = random.choice(bg_files)
    image = cv2.imread(bg_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    h, w = image.shape[:2]
    if h < IMG_SIZE or w < IMG_SIZE:
        scale = max(IMG_SIZE / h, IMG_SIZE / w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    transformed = bg_transform(image=image)
    return Image.fromarray(transformed['image']).convert("RGBA")

# --- START GENERATION ---
loop_cntr = 0
for i in tqdm(range(NUM_IMAGES_TO_GENERATE)):
    bg = generate_background()
    labels = []
    
    num_cubes = random.randint(0, 4) # Random number of cubes per image
    for _ in range(num_cubes):
        cube = random.choice(cubes).copy()
        
        # Random Transformations
        scale = random.uniform(0.8, 1.2) 
        angle = random.uniform(-45, 45)
        
        new_w, new_h = int(cube.width * scale), int(cube.height * scale)
        cube = cube.resize((new_w, new_h), Image.Resampling.LANCZOS)
        cube = cube.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        
        # Lighting Randomization
        cube = ImageEnhance.Brightness(cube).enhance(random.uniform(0.7, 1.3))
        cube = ImageEnhance.Contrast(cube).enhance(random.uniform(0.8, 1.2))
        
        # --- SHADOW LOGIC ---
        # 1. Extract and darken the alpha channel
        shadow_alpha = cube.split()[-1].point(lambda x: x * 0.4 if x > 0 else 0)
        
        # 2. Create solid black bands for R, G, and B (must be the same size as the cube)
        black_band = Image.new("L", cube.size, 0)
        
        # 3. Merge them correctly into an RGBA image
        shadow = Image.merge("RGBA", (black_band, black_band, black_band, shadow_alpha))
        
        # 4. Apply the blur
        shadow = shadow.filter(ImageFilter.GaussianBlur(5))

        # Placement logic
        max_x, max_y = IMG_SIZE - cube.width, IMG_SIZE - cube.height
        if max_x <= 0 or max_y <= 0: continue
            
        px, py = random.randint(0, max_x), random.randint(0, max_y)
        
        # Paste shadow first (offset slightly), then cube
        bg.paste(shadow, (px + 3, py + 3), shadow)
        bg.paste(cube, (px, py), cube)
        
        # YOLO Labeling
        bbox = cube.getbbox()
        if bbox:
            x_c = ((px + bbox[0] + px + bbox[2]) / 2.0) / IMG_SIZE
            y_c = ((py + bbox[1] + py + bbox[3]) / 2.0) / IMG_SIZE
            w = (bbox[2] - bbox[0]) / IMG_SIZE
            h = (bbox[3] - bbox[1]) / IMG_SIZE
            labels.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    
    # Convert the final composed PIL image (with cubes) back to an RGB numpy array
    final_composite_np = np.array(bg.convert("RGB"))
    
    # Apply global sensor noise and blur to EVERYTHING
    final_aug = global_transform(image=final_composite_np)
    final_img = Image.fromarray(final_aug['image'])

    if loop_cntr < int(0.8 * NUM_IMAGES_TO_GENERATE):
        split = 'train'
    else:
        split = 'val'
    loop_cntr += 1


    # Define dynamic paths
    img_path = os.path.join(DATASET_ROOT,'images', split, f"synth_{i}.jpg")
    lbl_path = os.path.join(DATASET_ROOT,'labels', split, f"synth_{i}.txt")

    # Save outputs (Convert RGBA back to RGB for JPEG compatibility)
    final_img.save(img_path, quality=90)
    with open(lbl_path, "w") as f:
        f.write("\n".join(labels))

print(f"\nDataset ready in {DATASET_ROOT}. Time to let that 4060 scream.")