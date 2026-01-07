import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.cm as cm    # Colormap을 위해 추가
from tqdm import tqdm
import cv2
import numpy as np

"""
일반 --> yolo
yolo --> 일반

중복박스 제거 함수
박스 크기 조정 함수 (실제 크기ㄴ 2배로 확대)


결과 그림 그려준다.
"""



def xml_to_yolo_bbox(bbox, w, h):
    """
    일반좌표계 -> yolo 좌표계
    bbox = [x1, y1, x2, y2]
    yolo_bbox = [x_center, y_center, width, height]
    """
    
    x_center = ((bbox[2]+bbox[0])/2)/w
    y_center = ((bbox[3]+bbox[1])/2)/h
    width = ((bbox[2]-bbox[0])/2)/w
    height = ((bbox[3]-bbox[1])/2)/h
    return [x_center, y_center, width, height]

def yolo_to_xml_bbox(bbox, w, h):
    """
    yolo 좌표계 -> 일반좌표계
    yolo_bbox = [x_center, y_center, width, height]
    bbox = [x1, y1, x2, y2]
    """

    x_min = (bbox[0]-bbox[2]) * w
    y_min = (bbox[1]-bbox[3]) * h
    x_max = (bbox[0]+bbox[2]) * w
    y_max = (bbox[1]+bbox[3]) * h
    return [x_min, y_min, x_max, y_max]



def select_highest_confidence_bbox(prediction):
    """
    중복박스 제거 함수
    신뢰도가 낮은 박스를 제거
    """
    
    if len(prediction) > 1:
        indices_to_remove = set()
        bboxes = prediction.boxes.xyxy #[p.boxes.xyxy.cpu() for p in prediction]
        confs = prediction.boxes.conf

        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if i in indices_to_remove or j in indices_to_remove:
                    continue

                box_i = bboxes[i]
                [x1_i, y1_i, x2_i, y2_i] = box_i
                center_x_i = (x1_i + x2_i) / 2
                center_y_i = (y1_i + y2_i) / 2

                box_j = bboxes[j]
                [x1_j, y1_j, x2_j, y2_j] = box_j
                center_x_j = (x1_j + x2_j) / 2
                center_y_j = (y1_j + y2_j) / 2

                is_center_i_in_j = (x1_j < center_x_i < x2_j) and (y1_j < center_y_i < y2_j)
                is_center_j_in_i = (x1_i < center_x_j < x2_i) and (y1_i < center_y_j < y2_i)

                if is_center_i_in_j or is_center_j_in_i:
                    # 신뢰도가 낮은 박스의 인덱스를 제거 목록에 추가
                    if confs[i] < confs[j]:
                        indices_to_remove.add(i)
                    else:
                        indices_to_remove.add(j)
        if indices_to_remove:
                # torch.bool 타입의 마스크 생성
            keep_mask = torch.ones(len(bboxes), dtype=torch.bool)
            for idx in indices_to_remove:
                keep_mask[idx] = False
            prediction_result = prediction[keep_mask]
            print(f"중첩 제거: {len(indices_to_remove)}개의 바운딩 박스가 제거되었습니다.")
            print(len(prediction), len(prediction_result))
            
        return prediction_result



def resize_bbox(prediction_result, scale_factor=2):
    """
    박스 크기 조정 함수 (실제 크기에 가깝게 2배로 확대)
    """

    scale_factor = 2 # 20% 확대
    r = prediction_result
    
    if len(r) > 0:   # result.boxes가 비어있지 않은 경우에만 크기 조절 실행 (중복박스 제거 후)
        modified_boxes_xyxy = r.boxes.xyxy.clone()
        confs = r.boxes.conf
        classes = r.boxes.cls
        n_boxes = []

        for i, box in enumerate(modified_boxes_xyxy):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            new_width = width * scale_factor
            new_height = height * scale_factor

            new_x1 = center_x - new_width / 2
            new_y1 = center_y - new_height / 2
            new_x2 = center_x + new_width / 2
            new_y2 = center_y + new_height / 2

            img_height, img_width = r.orig_shape
            new_x1 = max(0, new_x1).cpu().item()
            new_y1 = max(0, new_y1).cpu().item()
            new_x2 = min(img_width, new_x2).cpu().item()
            new_y2 = min(img_height, new_y2).cpu().item()

            n_boxes.append([int(new_x1), int(new_y1), int(new_x2), int(new_y2)])

        # result 객체의 bbox 정보를 수정된 정보로 교체
    new_result_bboxes = [[int(n[0].item()), n[1], n[2].item()] for n in zip(classes, n_boxes, confs)]
    return new_result_bboxes




def image_bbox_show_YOLO(img, bbox, fontsize, n_classes = 3, boxline_thickness=2, img_x_size=10, img_y_size=16):
    """
    bbox :
        [
            [[0, [3304, 634, 3422, 1284], 0.9730206727981567],
            [0, [5449, 665, 5571, 1221], 0.960870087146759],
            [0, [1883, 692, 1989, 1554], 0.909072756767273],
            [0, [1486, 708, 1622, 1815], 0.9034684896469116],
            [0, [801, 695, 924, 1734], 0.8977305293083191],
            [2, [4133, 2735, 6292, 5228], 0.8887861371040344],
            [1, [2776, 664, 2905, 1669], 0.8649317622184753],
            ...
        ]
    """
    # 각 class_number에 따른 색상 매핑 정의
    # 여기에 원하는 클래스 번호와 RGB 색상 튜플을 추가하세요.
    # 예시: {클래스_번호: (R, G, B)}
    # class_colors = {
    #     0: {"box": (255, 0, 0), "text": (255, 255, 0)},  # 클래스 0: 빨간색 박스, 노란색 텍스트
    #     1: {"box": (0, 255, 0), "text": (0, 0, 255)},  # 클래스 1: 초록색 박스, 파란색 텍스트
    #     2: {"box": (0, 0, 255), "text": (255, 0, 255)},  # 클래스 2: 파란색 박스, 마젠타 텍스트
    #     3: {"box": (255, 255, 0), "text": (0, 0, 0)},    # 클래스 3: 노란색 박스, 검은색 텍스트
    #     # 더 많은 클래스에 대한 색상을 추가할 수 있습니다.
    #     # 기본 색상 (만약 정의되지 않은 클래스 번호가 들어올 경우)
    #     "default": {"box": (128, 128, 128), "text": (255, 255, 255)} # 회색 박스, 흰색 텍스트
    # }
    color_map = cm.get_cmap('hsv', max(n_classes, 1))

    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV는 BGR, Matplotlib은 RGB이므로 변환

    fontpath = '/root/project/data_3090/jmahn/HANBatang.ttf'
    fontposition = 0.6 # 텍스트 위치 조정을 위한 비율

    # 이미지 처리를 위해 PIL Image로 변환
    image = Image.fromarray(img)

    for m in tqdm(bbox, total=len(bbox)):
        class_number = m[0]
        z = m[1]
        # p1, p2, p3, p4는 x1, y1, x2, y2를 의미
        x1, y1, x2, y2 = int(z[0]), int(z[1]), int(z[2]), int(z[3])
        prob = m[2] # 확률 (사용하지 않더라도 그대로 유지)

        text = str(class_number) # 박스 위에 표시할 텍스트 (클래스 번호)
        # class_number에 따라 색상 추출
        # Colormap에서 (R, G, B, A) 값을 얻고, 0-255 범위로 변환
        # n_classes가 0이 아닐 때만 정규화

        normalized_class_number = class_number / (n_classes -1) if n_classes > 1 else 0   # 0 <= normalized_class_number <= 1
        rgba_color = color_map(normalized_class_number)
        select_color = (int(rgba_color[0]*255), int(rgba_color[1]*255), int(rgba_color[2]*255))
        # 폰트 크기 계산
        # 박스 너비에 비례하여 폰트 크기 조절
        font_size = max(int((x2 - x1) * fontsize), 10) # 최소 폰트 크기 10
        font = ImageFont.truetype(fontpath, font_size)

        draw = ImageDraw.Draw(image)

        # 텍스트 위치 조정 (박스 위에 표시)
        # 텍스트의 높이를 고려하여 y1에서 텍스트 크기만큼 위로 이동
        # text_width, text_height = draw.textsize(text, font=font)
        # text_position = (x1, y1 - text_height - 5) # 5는 여백

        # 이미지 경계 체크: 텍스트가 이미지 위로 넘어가지 않도록 조정
        # if text_position[1] < 0:
        #     text_position = (x1, y1 + 5) # 만약 위로 넘어가면 박스 아래에 표시

        # draw.text(text_position, text, font=font, fill=select_color)

        # 박스 그리기 (PIL Image를 다시 NumPy 배열로 변환 후 OpenCV 사용)
        # OpenCV는 RGB 순서로 색상을 받으므로, PIL에서 설정된 색상 튜플을 그대로 사용.
        # draw.rectangle은 PIL ImageDraw의 메서드이므로 PIL Image에 직접 그림
        draw.rectangle([(x1, y1), (x2, y2)], outline=select_color, width=boxline_thickness)

    # 모든 드로잉이 끝난 후 NumPy 배열로 최종 변환
    imgs = np.array(image)

    plt.figure(figsize=(img_x_size, img_y_size))
    print(imgs.shape)
    plt.imshow(imgs)
    plt.axis('off') # 축 정보 제거
    plt.savefig('result.png')
    plt.show()



if __name__ == "__main__":
    image_bbox_show_YOLO(img=img, 
                     bbox=new_result_bboxes, 
                     fontsize=10, n_classes = 3, boxline_thickness=8, 
                     img_x_size=20, img_y_size=20)