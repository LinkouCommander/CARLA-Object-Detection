# CARLA Object Detection using YOLOv8
This Repository contains a CARLA Object Detection model trained by YOLOv8. Classes included: 
bike,
motobike,
person,
traffic_light_green,
traffic_light_orange,
traffic_light_red,
traffic_sign_30,
traffic_sign_60,
traffic_sign_90,
vehicle

## Train
Run 'train.py'. 
Don't forget to modify path of train, test, and valid in data.yaml.

## Predict
Run 'predict.py'. 
Don't forget to put 'best.pt', image you want to predict , and 'predict.py' in same path.
'best.pt' located in ./run/detect/train3/weights