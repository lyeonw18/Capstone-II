import os
import cv2
import torch
import random
import albumentations as A
from tqdm import tqdm
from torchvision import models, transforms as T
from PIL import Image, ImageEnhance


# 1. 밝기 분류기 로드
def load_brightness_classifier(model_path, device):
    model = models.resnet18(weights=None)  # 구조만 로드
    model.fc = torch.nn.Linear(model.fc.in_features, 3)  # s1/s3/s5
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model



# 2. 이미지 → 밝기 단계 예측
def predict_stage(model, img, device):
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
    return ["s1", "s3", "s5"][pred]

# 3. 단계별 밝기 증강
def brightness_stage_augment(img, stage):

    def gamma(img, g):
        inv = 1.0 / g
        table = [int((i / 255.0) ** inv * 255) for i in range(256)]
        if img.mode == "RGB":
            table = table * 3
        return img.point(table)

    # Stage 1: 밝게 (저조도 보완 방향)
    if stage == "s1":
        img = gamma(img, random.uniform(1.05, 1.12))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1.00, 1.08))

    # Stage 3: 중간 단계 (밝기 변화 최소 + 색감 변화)
    elif stage == "s3":
        img = gamma(img, random.uniform(0.98, 1.02))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.95, 1.05))

    # Stage 5: 저조도 단계 (안전한 어둡기)
    else:
        img = gamma(img, random.uniform(0.92, 0.98))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.92, 1.02))

    return img



# 4. YOLO용 기본 증강
def get_yolo_augment():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=7, p=0.5),
        A.Resize(640, 640)
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.2,
        clip=True
    ))


# 5. 실행 파이프라인 (train만 수행)
if __name__ == "__main__":

    # ===== 입력 경로 =====
    src_img_dir = r"E:/aihub_lowlight/balanced_scene_split/train/images"
    src_lbl_dir = r"E:/aihub_lowlight/balanced_scene_split/train/labels"

    # ===== 출력 경로 =====
    dst_img_dir = r"E:/aihub_lowlight/dataset/brightness_stage_aug_1202/train/images"
    dst_lbl_dir = r"E:/aihub_lowlight/dataset/brightness_stage_aug_1202/train/labels"

    # 분류기 경로
    model_path = r"E:/aihub_lowlight/brightness_stage_classifier.pt"

    # 출력 폴더 생성
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    # 장치 선택
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"현재 장치: {device}")

    # 모델/증강기 불러오기
    model = load_brightness_classifier(model_path, device)
    yolo_tf = get_yolo_augment()

    
    # 이미지 단일 폴더에서 바로 처리 
    print("\n brightness augmentation 실행 중...")

    for fname in tqdm(os.listdir(src_img_dir), desc="Train 증강 진행 중"):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(src_img_dir, fname)
        lbl_path = os.path.join(src_lbl_dir, fname.replace(".jpg", ".txt"))
        dst_img_path = os.path.join(dst_img_dir, fname)
        dst_lbl_path = os.path.join(dst_lbl_dir, fname.replace(".jpg", ".txt"))

        # 이미지 로드
        image = cv2.imread(img_path)
        if image is None or not os.path.exists(lbl_path):
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 라벨 읽기
        with open(lbl_path, "r") as f:
            lines = [x.strip().split() for x in f.readlines() if len(x.strip()) > 0]
        if len(lines) == 0:
            continue

        bboxes = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in lines]
        class_labels = [int(x[0]) for x in lines]

        # YOLO 증강
        transformed = yolo_tf(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = transformed['image']
        aug_boxes = transformed['bboxes']
        aug_labels = transformed['class_labels']

        # PIL 변환
        pil_img = Image.fromarray(aug_img)

        # 밝기 단계 예측 + 증강
        stage = predict_stage(model, pil_img, device)
        pil_img = brightness_stage_augment(pil_img, stage)

        # 저장
        pil_img.save(dst_img_path)

        with open(dst_lbl_path, "w") as f:
            for cls, (x, y, w, h) in zip(aug_labels, aug_boxes):
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("\n brightness_stage_aug_1206 Train 증강 완료")
    print("이미지:", dst_img_dir)
    print("라벨:", dst_lbl_dir)
