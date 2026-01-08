# 홍루몽 YOLO Layout Parser
Ultralytics YOLOv8 기반으로 홍루몽 고서 이미지의 레이아웃(G1/G2/G3)을 추출하는 프로젝트입니다. 간단한 추론 스크립트와 FastAPI 엔드포인트로 서비스화할 수 있도록 구성되었습니다.

- FastAPI + API Key(`x-api-key`) 인증, `GET /layout` 엔드포인트 제공
- YOLO 추론 결과에서 중첩 박스 제거 후 2배 확장하여 실제 인쇄 영역과 비슷하게 보정
- 시각화 유틸리티로 결과를 `result.png`로 저장
- 기본 가중치: `/root/project/data/jmahn/data/layout_parsing/홍루몽/yolo/train/test/img_resize_1120_batch_4_epoch_202/weights/best.pt`

## Repository Map
- `src/main_layout_api.py` : FastAPI 앱, API Key 검증 및 `/layout` 서비스
- `src/models.py` : YOLO 가중치 로더
- `src/modules.py` : YOLO 추론 래퍼
- `src/utils.py` : 중복 박스 제거(`select_highest_confidence_bbox`), 2배 확장(`resize_bbox`), 시각화(`image_bbox_show_YOLO`)
- `infer.py` : 단일 이미지 추론 예제
- `train/` : 학습 산출물(가중치) 보관 디렉터리

## 환경 준비
### Docker (cuda12.2)
```bash
docker pull pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
docker run --gpus all --shm-size 32gb --ipc=host --name ocr_api_layout \
  -v /data:/root/project/data -p 8809:8809 \
  -it pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime /bin/bash
```

### 로컬/컨테이너 공통 의존성
필수 패키지 예시:
```bash
pip install ultralytics fastapi uvicorn opencv-python pillow matplotlib tqdm requests
```
PyTorch는 CUDA 12.x 지원 빌드를 사용하세요.

## FastAPI 서비스 실행
1) `src/main_layout_api.py` 내 경로/디바이스를 환경에 맞게 수정  
   - `API_KEY`: 기본값을 서비스 키로 교체  
   - `device`: 기본 `cuda:2` → 사용 가능한 GPU로 변경  
   - `model_path`: 배포용 가중치 경로  
2) 실행
```bash
uvicorn src.main_layout_api:app --host 0.0.0.0 --port 8809
# 또는 gunicorn: gunicorn src.main_layout_api:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8809 --timeout 600
```

### 요청 예시
`x-api-key` 헤더가 필수입니다.
```bash
curl -H "x-api-key: <YOUR_KEY>" \
  "http://localhost:8809/layout?url=https://example.com/sample.jpg"
```
로컬 파일 경로도 `url` 파라미터에 전달할 수 있습니다.

### 응답 포맷
`/layout`는 중복 제거 및 확장된 박스를 다음 형식으로 반환합니다.
```json
{
  "result": [
    {
      "id": "82f9...2981",
      "mark": { "x": 1444.0, "y": 877.7, "width": 144.6, "height": 458.9 },
      "comment": "0",
      "index": "",
      "group": "G1"
    }
  ]
}
```
클래스 매핑: `0 -> G1`, `1 -> G2`, `2 -> G3`.

## Python 추론 예제
```python
from src.models import yolo_model_load
from src.modules import yolo_model_infer
from src.utils import select_highest_confidence_bbox, resize_bbox, image_bbox_show_YOLO

img_path = "/path/to/image.jpg"
model_path = "/root/project/data/jmahn/data/layout_parsing/홍루몽/yolo/train/test/img_resize_1120_batch_4_epoch_202/weights/best.pt"
device = "cuda:0"

model = yolo_model_load(model_path)
pred = yolo_model_infer(model, img_path=img_path, device=device, conf=0.3)
filtered = select_highest_confidence_bbox(pred)          # 중심 포함 관계로 중첩 제거
scaled = resize_bbox(filtered)                           # 박스 2배 확장 + 이미지 경계 클램핑
image_bbox_show_YOLO(img=img_path, bbox=scaled, fontsize=0.01, n_classes=3)
```
`image_bbox_show_YOLO`는 `result.png`를 저장하며, 폰트 경로가 기본 `/root/project/data_3090/jmahn/HANBatang.ttf`로 설정되어 있으니 필요시 수정하세요.

## 후처리 로직 요약
- `select_highest_confidence_bbox`: 박스 중심이 다른 박스 내부에 있는 경우 신뢰도가 낮은 박스를 제거
- `resize_bbox`: 각 박스를 중심 기준 2배로 확장한 뒤 이미지 경계에 맞춰 클램프
- `formatter_layout`: API 응답 형태로 변환하며 UUID를 생성하고 클래스명(G1/G2/G3)을 포함

## 모델/데이터 위치
- 추론 기본 가중치: `/root/project/data/jmahn/data/layout_parsing/홍루몽/yolo/train/test/img_resize_1120_batch_4_epoch_202/weights/best.pt`
- TensorRT 엔진(옵션): `/root/project/layout_parsing/홍루몽/yolo/train/img_resize_1120_batch_4_epoch_202/weights/best.engine`

## 팁
- API Key는 운영 환경에서 반드시 교체하고, 컨테이너/배포 환경 변수를 활용해 관리하세요.
- 다른 GPU 번호를 사용할 경우 `device` 값을 맞춰주고, 모델 경로에 접근 권한이 있는지 확인하세요.
- 추론 입력이 매우 클 경우 `segment_spilt_size`와 `fold_size`(API 코드 상단)를 조정해 타일 크기를 변경할 수 있습니다.
