def yolo_model_infer(model, img_path, device='cuda:3', conf=0.3):
    prediction = model.predict(img_path, device=device, conf=conf)[0]
    return prediction