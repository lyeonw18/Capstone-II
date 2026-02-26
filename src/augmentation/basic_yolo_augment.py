import os, cv2, random, numpy as np
import albumentations as A
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

# YOLO bbox-aware basic_augment
basic_tf = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
        A.Resize(640, 640)
    ],
    bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.2,
        clip=True
    )
)


# route settings
src_img_root = "E:/aihub_lowlight/balanced_scene_split/train/images"
src_lbl_root = "E:/aihub_lowlight/balanced_scene_split/train/labels"

dst_img_dir = "E:/aihub_lowlight/dataset/basic_aug/train/images"
dst_lbl_dir = "E:/aihub_lowlight/dataset/basic_aug/train/labels"


os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(dst_lbl_dir, exist_ok=True)

# Original image list
img_files = [f for f in os.listdir(src_img_root) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

count = 0
for fname in tqdm(img_files, desc="기본 증강 중"):
    img_path = os.path.join(src_img_root, fname)
    lbl_path = os.path.join(src_lbl_root, fname.rsplit(".", 1)[0] + ".txt")

    if not os.path.exists(lbl_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    # label reading
    with open(lbl_path, "r") as f:
        raw = [x.strip().split() for x in f.readlines() if x.strip()]
    if len(raw) == 0:
        continue

    class_labels = [int(x[0]) for x in raw]
    bboxes = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in raw]

    # augmentation
    try:
        transformed = basic_tf(image=image, bboxes=bboxes, class_labels=class_labels)
    except Exception as e:
        print(f"⚠{fname} 증강 오류: {e}")
        continue

    new_boxes = transformed["bboxes"]
    new_labels = transformed["class_labels"]

    # Leave only valid objects
    if len(new_boxes) == 0:
        continue

    # Change file name (to prevent overwriting)
    base = fname.rsplit(".", 1)[0]
    out_img = os.path.join(dst_img_dir, f"{base}_aug.jpg")
    out_lbl = os.path.join(dst_lbl_dir, f"{base}_aug.txt")

    # Save BGR output without color conversion
    cv2.imwrite(out_img, transformed["image"])

    with open(out_lbl, "w") as f:
        for cls, (x, y, w, h) in zip(new_labels, new_boxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    count += 1

print(f"\n YOLO 기본 증강 완료: {count}장 생성!")
print(" 저장 위치:", dst_img_dir)

