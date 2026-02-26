import os
import shutil

base = r"E:\aihub_lowlight\balanced_scene_split"
out_base = r"E:\aihub_lowlight\balanced_scene_split\brightness_stage_split"

splits = ["train", "val", "test"]
stages = ["stage1", "stage3", "stage5"]

# 출력 구조 생성 (ImageFolder와 호환)
for split in splits:
    for stage in stages:
        os.makedirs(os.path.join(out_base, split, stage), exist_ok=True)

def get_stage(fname):


    parts = fname.split("_")
    if len(parts) < 4:
        return None

    last = parts[3]  # L04A / L06C / L07E 등

    if "A" in last:
        return "stage1"
    elif "C" in last:
        return "stage3"
    elif "E" in last:
        return "stage5"
    return None


# 분류 시작
for split in splits:
    img_dir = os.path.join(base, split, "images")   # 수정된 경로
    print(f"\n{split.upper()} 처리 중...")

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(".jpg"):
            continue

        stage = get_stage(fname)
        if stage is None:
            print("스테이지 정보 없음:", fname)
            continue

        src = os.path.join(img_dir, fname)
        dst = os.path.join(out_base, split, stage, fname)

        shutil.copy2(src, dst)

    print(f"{split} 완료")

