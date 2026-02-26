import os
import json
from glob import glob
from tqdm import tqdm


# 경로 설정
json_dir = r"E:\aihub_lowlight\028.저조도_환경_데이터\01-1.정식개방데이터\Training\02.라벨링데이터\extracted"
out_label_dir = r"E:\aihub_lowlight\028.저조도_환경_데이터\01-1.정식개방데이터\Training\02.라벨링데이터\yolo_labels"

os.makedirs(out_label_dir, exist_ok=True)


# JSON 파일 목록
json_files = glob(os.path.join(json_dir, "**", "*.json"), recursive=True)
print("총 JSON 파일 개수:", len(json_files))


# 클래스 이름 → 숫자 ID 매핑
# 등장 횟수 저장 기능 추가
class_map = {}
class_count = {}


# JSON → YOLO txt 변환 함수
def json_to_yolo(json_path):
    base = os.path.splitext(os.path.basename(json_path))[0]

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = data["Raw_Data_Info."]["Resolution"].replace(" ", "")
    w, h = map(int, res.split(","))

    anns = data["Learning_Data_Info."].get("Annotations", [])
    yolo_lines = []

    for ann in anns:
        if ann.get("Type") != "Bounding_box":
            continue

        cls_name = ann["Class_ID"]

        # class_map 자동 생성
        if cls_name not in class_map:
            class_map[cls_name] = len(class_map)

        # 등장 횟수 기록
        class_count[cls_name] = class_count.get(cls_name, 0) + 1

        cls_id = class_map[cls_name]

        x1, y1, x2, y2 = ann["Type_value"]

        # YOLO 변환
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # txt 저장
    out_path = os.path.join(out_label_dir, base + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))


# 전체 JSON 변환 실행
for jp in tqdm(json_files, desc="Converting JSON → YOLO"):
    json_to_yolo(jp)

print("\n YOLO 라벨 생성 완료")
print("저장 위치:", out_label_dir)


# 클래스 정보 출력
print("\n최종 클래스 목록(Class_ID)")
print("총 클래스 수:", len(class_map))

# ID 기준 정렬 출력
sorted_map = sorted(class_map.items(), key=lambda x: x[1])

for name, idx in sorted_map:
    print(f"{idx}: {name}")

# 클래스 등장 횟수 출력
print("\n클래스 등장 횟수")
for name, count in sorted(class_count.items(), key=lambda x: x[0]):
    print(f"{name}: {count}개")
