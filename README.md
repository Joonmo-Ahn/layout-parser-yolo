# 홍루몽 문서 레이아웃 파싱 (YOLO 기반)

고문서 **홍루몽(Hongloumeng)** 페이지의 레이아웃을 분석하기 위한 프로젝트입니다. 현재는 YOLO 기반 객체 검출로 영역(G1/G2/G3)을 분할하며, 향후 현대 문서와 다른 모델(예: Detectron2, PP-YOLOE, LayoutLM 계열 등)로 확장을 염두에 두고 있습니다.

## 프로젝트 개요
- 목표: 고문서 페이지에서 문단/블록(G1, G2, G3) 영역을 자동 검출·시각화.
- 현 단계: YOLO 모델을 활용한 추론 파이프라인 및 시각화 유틸리티 정비.
- 확장 계획: 현대 문서 레이아웃, 추가 백본/헤드, 멀티 모델 지원.

## 파일 구조
- `infer.py`: 단일 스크립트 엔트리포인트. 하단의 경로 변수만 수정하면 바로 실행.
- `infer.ipynb`: 순차적인 셀로 끝나는 빠른 추론 노트북. (일반모델, TensorRT모델)
- `infer_step_by_step.ipynb`: 코드 단계와 중간 시각화를 포함한 디버깅용 느린 버전.
- `yolo11_홍루몽.ipynb`: 초기 YOLOv11 학습/평가 실험 기록.
- `src/models.py`: `ultralytics.YOLO` 모델 로더 래퍼.
- `src/modules.py`: `model.predict` 추론 래퍼(device/conf 설정).
- `src/utils.py`: `xml↔yolo` 변환, 박스 중복 제거·확장, `result.png` 시각화.
- `train/test/img_resize_1120_batch_4_epoch_202/`: 홍루롱 레이아웃 학습 산출물(플롯, 혼동행렬, 다양한 포맷의 가중치).

## 빠른 시작 (YOLO 추론)
1. **의존성 설치** (Python 3.11+, GPU 환경에 맞는 Torch/CUDA 선택):
   ```bash
   pip install ultralytics --no-deps         
   pip install torch torchvision torchaudio  # CUDA 빌드 선택 
   pip install torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/cu124 # cuda 12.4/torch 2.6.0 에 맞게 설치
   pip install opencv-python pillow matplotlib tqdm pandas
   ```
2. **경로 설정** (`infer.py` 하단의 변수 또는 환경변수로 교체):
   ```bash
   IMG_PATH="/root/project/data/vision/data/nara/홍루몽/홍루몽1-6권/layout/images/K4-6864_001_紅樓夢_홍루몽(54)_(1)_0031.jpg"
   MODEL_PATH="/root/project/data/vision/data/layout/hong_ru_mong/checkpoints/img_resize_1120_batch_4_epoch_202/weights/best.pt"
   DEVICE="cuda:0"  # GPU가 없으면 "cpu"
   ```
3. **실행**:
   ```bash
   python infer.py
   ```
   - `infer.py`에서 `img_path`, `model_path`, `device`를 수정 후 실행합니다.
   - 스크립트는 겹치는 박스를 제거하고 약간 확장한 뒤 `result.png`를 저장합니다.

## 클래스 라벨 (현재 YOLO 헤드)
- `0`: G1                     
- `1`: G2
- `2`: G3

## 노트북 사용 팁
- 대화형/디버그가 필요하면 `infer_step_by_step.ipynb`를, 가장 빠른 확인은 `infer.ipynb`를 사용.
- `yolo11_홍루몽.ipynb`는 학습 실험 로그이며, 주요 하이퍼파라미터는 `train/test/img_resize_1120_batch_4_epoch_202/args.yaml`에 기록돼 있습니다.
- 커밋 전 노트북 출력은 지워 용량을 줄이세요.

## 모델 산출물
- 지표/플롯: `results.png`, `confusion_matrix*.png`, `PR_curve.png` 등
- 내보내기: `weights/best.pt`, `best.onnx`, `best.engine`, `last.pt`
- 대용량 산출물은 Git에 올리지 않도록 `.gitignore_cursor`에서 관리합니다.

## 시각화/폰트 주의
`src/utils.py`는 기본 한글 폰트를 `/root/project/data_3090/jmahn/HANBatang.ttf`로 가정합니다. 폰트 위치가 다르면 `fontpath`를 바꿔 실행 오류를 방지하세요.

## 확장 아이디어
- 헬퍼(`xml_to_yolo_bbox`, `yolo_to_xml_bbox`, `image_bbox_show_YOLO`)는 모델 불문 공용입니다. 다른 감지기나 레이아웃 파서로 교체할 때 재사용할 수 있습니다.
- `infer.py`를 CLI 인자(`--device`, `--conf`, `--model_path`, `--fontpath`)로 일반화하면 다모델 실험에 유용합니다.

## 데이터 취급
원본 이미지·XML/라벨은 저장소에 포함되지 않습니다. 실행 시 `IMG_PATH`/`MODEL_PATH`를 로컬 또는 마운트된 데이터 경로로 지정하세요.

## 깃허브 업로드 가이드
- 코드, 설정, 소형 샘플만 유지하고 학습된 가중치·대형 플롯·생성물은 제외합니다.
- 노트북 출력은 지워서 용량을 최소화하고, 실험 결과는 외부 스토리지에 보관하세요.

