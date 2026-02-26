import os
import json
import re
import random
from collections import defaultdict
from shutil import copy2

# 0. 경로 설정
json_dir = r"E:\aihub_lowlight\028.저조도_환경_데이터\01-1.정식개방데이터\Training\02.라벨링데이터\extracted"
img_dir = r"E:\aihub_lowlight\028.저조도_환경_데이터\01-1.정식개방데이터\Training\01.원천데이터"
label_yolo_dir = r"E:\aihub_lowlight\028.저조도_환경_데이터\01-1.정식개방데이터\Training\02.라벨링데이터\yolo_labels"

out_base = r"E:\aihub_lowlight\balanced_scene_split_no_rare"
os.makedirs(out_base, exist_ok=True)
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(out_base, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_base, split, "labels"), exist_ok=True)


# 1. JSON에서 클래스 카운트 계산
class_counts = defaultdict(int)
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

for jf in json_files:
    with open(os.path.join(json_dir, jf), "r", encoding="utf8") as f:
        data = json.load(f)

    for ann in data["Learning_Data_Info."]["Annotations"]:
        class_counts[ann["Class_ID"]] += 1

# 300개 이상 등장한 클래스만 candidate
valid_classes = {cls for cls, cnt in class_counts.items() if cnt > 300}
print("[300개 이상 등장 클래스 수]:", len(valid_classes))
print(valid_classes)


# 2. filename → date & frame 번호 추출
def parse_filename(name):
    # 날짜
    date_match = re.search(r'_(\d{6})_', name)
    if not date_match:
        return None, None
    date = date_match.group(1)

    # 끝부분 숫자를 frame으로 사용
    frame_match = re.search(r'(\d+)$', name)
    if not frame_match:
        return date, None

    return date, int(frame_match.group(1))


# 3. valid JSON 목록 필터링
valid_json = []

for jf in json_files:
    with open(os.path.join(json_dir, jf), "r", encoding="utf8") as f:
        data = json.load(f)

    anns = data["Learning_Data_Info."]["Annotations"]
    if any(ann["Class_ID"] in valid_classes for ann in anns):
        valid_json.append(jf)

print("[유효 JSON 수]:", len(valid_json))


# 4. 날짜 + frame 기반 scene grouping
info_list = []
for jf in valid_json:
    base = jf.replace(".json", "")
    date, frame = parse_filename(base)
    if date is None or frame is None:
        continue
    info_list.append((jf, date, frame))

# 날짜별 grouping
date_groups = defaultdict(list)
for jf, date, frame in info_list:
    date_groups[date].append((jf, frame))

scene_groups = []
for date, items in date_groups.items():
    items = sorted(items, key=lambda x: x[1])
    visited = set()

    for jf, frame in items:
        if jf in visited:
            continue

        group = []
        for jf2, f2 in items:
            if abs(f2 - frame) <= 4:
                group.append(jf2)
                visited.add(jf2)

        scene_groups.append(group)

print("[Scene 그룹 수]:", len(scene_groups))


# 5. scene당 최대 30장 + 클래스당 최대 250장 확보
MAX_SCENE_PICK = 30
MAX_PER_CLASS = 250

cls_to_images = defaultdict(list)

for group in scene_groups:
    random.shuffle(group)
    selected = []
    used_locally = set()

    # 1) scene 안에서 클래스 다양성 우선
    for jf in group:
        if len(selected) >= MAX_SCENE_PICK:
            break

        with open(os.path.join(json_dir, jf), "r", encoding="utf8") as f:
            data = json.load(f)

        anns = data["Learning_Data_Info."]["Annotations"]
        img_classes = [ann["Class_ID"] for ann in anns if ann["Class_ID"] in valid_classes]
        if not img_classes:
            continue

        main_cls = img_classes[0]

        if main_cls in used_locally:
            continue
        if len(cls_to_images[main_cls]) >= MAX_PER_CLASS:
            continue

        selected.append(jf)
        cls_to_images[main_cls].append(jf)
        used_locally.add(main_cls)

    # 2) 남은 자리 클래스 채우기
    for jf in group:
        if len(selected) >= MAX_SCENE_PICK:
            break

        with open(os.path.join(json_dir, jf), "r", encoding="utf8") as f:
            data = json.load(f)

        anns = data["Learning_Data_Info."]["Annotations"]
        img_classes = [ann["Class_ID"] for ann in anns if ann["Class_ID"] in valid_classes]
        if not img_classes:
            continue

        main_cls = img_classes[0]

        if len(cls_to_images[main_cls]) >= MAX_PER_CLASS:
            continue

        selected.append(jf)
        cls_to_images[main_cls].append(jf)


# 6. 부족 클래스(<50장) 제거
RARE_THRESHOLD = 50
rare_classes = {cls for cls, lst in cls_to_images.items() if len(lst) < RARE_THRESHOLD}

print("\n 제거할 소수 클래스:", rare_classes)

for cls in rare_classes:
    del cls_to_images[cls]

print("제거 후 클래스 수:", len(cls_to_images))


# 7. train/val/test split 생성
split_data = {"train": [], "val": [], "test": []}

for cls, imgs in cls_to_images.items():
    random.shuffle(imgs)
    imgs = imgs[:MAX_PER_CLASS]

    n = len(imgs)
    t = int(n * 0.7)
    v = int(n * 0.15)

    split_data["train"] += imgs[:t]
    split_data["val"] += imgs[t:t+v]
    split_data["test"] += imgs[t+v:]

print("\n[최종 split 크기]")
for k in split_data:
    print(k, len(split_data[k]))


# 8. 이미지/라벨 복사
def find_image(base):
    for root, _, files in os.walk(img_dir):
        for ext in [".jpg", ".jpeg", ".png"]:
            path = os.path.join(root, base + ext)
            if os.path.exists(path):
                return path
    return None

def copy_pairs(jf_list, split):
    for jf in jf_list:
        base = jf.replace(".json", "")
        img_path = find_image(base)
        lbl_path = os.path.join(label_yolo_dir, base + ".txt")

        if img_path and os.path.exists(lbl_path):
            copy2(img_path, os.path.join(out_base, split, "images"))
            copy2(lbl_path, os.path.join(out_base, split, "labels"))

copy_pairs(split_data["train"], "train")
copy_pairs(split_data["val"], "val")
copy_pairs(split_data["test"], "test")

