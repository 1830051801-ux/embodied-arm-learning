import json
from pathlib import Path


SUPPORTED_CLASSES = {"cup", "bottle", "box"}


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def adapt_raw_yolo(raw):
    standard = {
        "image_size": [raw["width"], raw["height"]],
        "detections": [],
    }
    for box in raw["boxes"]:
        class_name = box["label"]
        if class_name not in SUPPORTED_CLASSES:
            continue
        standard["detections"].append(
            {
                "class_name": class_name,
                "confidence": float(box["score"]),
                "bbox": [float(v) for v in box["xyxy"]],
            }
        )
    return standard


def choose_target(standard):
    detections = standard["detections"]
    if not detections:
        raise ValueError("no supported detections")
    return max(detections, key=lambda d: d["confidence"])


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def main():
    base_dir = Path(__file__).parent
    raw_path = base_dir / "day9_raw_yolo_output.json"
    standard_path = base_dir / "day9_standard_yolo_detections.json"

    print_header("Step 0: Why use an adapter?")
    print("Different YOLO libraries may output different field names.")
    print("The robot pipeline should use one stable standard JSON format.")

    print_header("Step 1: Load raw YOLO-like output")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    print("raw file:", raw_path)
    print(json.dumps(raw, indent=2))

    print_header("Step 2: Convert raw output to standard detection JSON")
    standard = adapt_raw_yolo(raw)
    standard_path.write_text(json.dumps(standard, indent=2), encoding="utf-8")
    print("standard file:", standard_path)
    print(json.dumps(standard, indent=2))

    print_header("Step 3: Choose target and compute bbox center")
    target = choose_target(standard)
    center = bbox_center(target["bbox"])
    print("chosen target:", target)
    print(f"bbox_center=({center[0]:.1f}, {center[1]:.1f})")
    print("Meaning: this is ready for Day 8 calibration and PyTorch policy input.")

    print_header("Step 4: Meaning for real YOLO")
    print("If you use ultralytics YOLO, convert its boxes to this same standard JSON.")
    print("After this adapter, Day 10/robot command scripts do not care which YOLO library produced it.")


if __name__ == "__main__":
    main()

