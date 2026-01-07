from ultralytics import YOLO


def yolo_model_load(model_path):
    model = YOLO(model_path)
    return model




