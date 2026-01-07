from src.models import yolo_model_load       # 모델 볼러온다.
from src.modules import yolo_model_infer      # 모델 추론한다.
from src.utils import image_bbox_show_YOLO, select_highest_confidence_bbox, resize_bbox
import time







if __name__ == "__main__":
    img_path = '/root/project/data/vision/data/nara/홍루몽/홍루몽1-6권/layout/images/K4-6864_001_紅樓夢_홍루몽(54)_(1)_0031.jpg'

    model_path = '/root/project/data/vision/data/layout/hong_ru_mong/checkpoints/img_resize_1120_batch_4_epoch_202/weights/best.pt'
    model = yolo_model_load(model_path=model_path)
    start = time.time()
    prediction = yolo_model_infer(model, img_path=img_path, device=device, conf=0.3)
    prediction_result = select_highest_confidence_bbox(prediction)
    resize_bbox_result = resize_bbox(prediction_result)     # [(classes, n_boxes, confs), .... ]
    final = time.time()
    print(f'추론 종료: {final-start:.3f}s')
    image_bbox_show_YOLO(img=img_path, 
                     bbox=resize_bbox_result, 
                     fontsize=10, n_classes = 3, boxline_thickness=8, 
                     img_x_size=20, img_y_size=20)
    """
    classes
    0: G1
    1: G2
    2: G3
    """
