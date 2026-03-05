from ultralytics import YOLO

# 1. Load your newly minted model
# Note: If you ran the training command multiple times, YOLO creates new folders 
# like train2, train3, etc. Make sure this points to your most recent run.
model_path = r"runs\detect\train\weights\best.pt"
model = YOLO(model_path)

# 2. Point it at your raw test image
test_image = "raw_1772479609.jpg" 

# 3. Run inference
# conf=0.5 means it only draws a box if it's at least 50% sure it's a cube
print("Running inference...")
results = model.predict(source=test_image, conf=0.5, show=True, save=True)
print(type(results))
print(results)

print("Done. If the window didn't pop up, check the 'runs/detect/predict' folder for the saved image.")