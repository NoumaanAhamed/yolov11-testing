from ultralytics import YOLO

model = YOLO('./best.pt')  # Load model

# Run inference on a single image
results = model.predict(source='testvid.mp4', save=True)  # save plotted images

# Image is saved to './runs/detect/exp/test.jpg'