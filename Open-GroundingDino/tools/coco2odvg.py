import argparse
import jsonlines
from tqdm import tqdm
import json
from pycocotools.coco import COCO
from collections import defaultdict
import os

id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
key_list = list(id_map.keys())
val_list = list(id_map.values())

def coco_to_xyxy(bbox):
    x, y, width, height = bbox
    return [round(x, 2), round(y, 2), round(x + width, 2), round(y + height, 2)]

def main(args):
    # Load dataset split
    with open(args.split, "r") as f:
        split_data = json.load(f)

    # Create sets of image filenames
    train_images_set = set([img for cat in split_data["train"].values() for img in cat])
    test_images_set = set([img for cat in split_data["test"].values() for img in cat])

    # Load COCO
    coco = COCO(args.input)
    cats = coco.loadCats(coco.getCatIds())
    nms = {cat['id']: cat['name'] for cat in cats}

    # Prepare containers
    train_metas = []
    test_images = []
    test_annotations = []
    ann_by_image = defaultdict(list)

    # Map annotations by image_id
    for ann in coco.dataset["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    # Process images
    for img in tqdm(coco.dataset["images"], desc="Processing images"):
        fname = img["file_name"]
        fname_base = os.path.basename(img["file_name"])
        img_id = img["id"]
        
        # Train check
        if fname_base in train_images_set:
            # Convert to ODV-G
            instance_list = []
            for ann in ann_by_image[img_id]:
                bbox_xyxy = coco_to_xyxy(ann["bbox"])
                label = ann["category_id"]
                category = nms[label]
                ind = val_list.index(label)
                label_trans = key_list[ind]
                attributes = ann.get("attributes", [])
                instance_list.append({
                    "bbox": bbox_xyxy,
                    "label": label_trans,
                    "category": category,
                    "attributes": attributes
                })
            train_metas.append({
                "filename": fname,
                "height": img["height"],
                "width": img["width"],
                "detection": {"instances": instance_list}
            })
        # Test check
        elif fname_base in test_images_set:
            # Keep COCO format
            test_images.append(img)
            test_annotations.extend(ann_by_image[img_id])

    # Save train ODV-G
    with jsonlines.open(args.train_output, mode="w") as writer:
        writer.write_all(train_metas)

    # Save test COCO
    test_coco_data = {
        "info": {},
        "licenses": [],
        "images": test_images,
        "annotations": test_annotations,
        "categories": coco.dataset["categories"]
    }
    with open(args.test_output, "w") as f:
        json.dump(test_coco_data, f)

    print(f"Done. Train ODV-G: {args.train_output}, Test COCO: {args.test_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Split COCO to ODV-G for train and COCO for test.")
    parser.add_argument("--input", "-i", required=True, type=str, help="Input COCO JSON")
    parser.add_argument("--split", "-s", required=True, type=str, help="Dataset split JSON")
    parser.add_argument("--train_output", default="train_odvg.jsonl", type=str)
    parser.add_argument("--test_output", default="test_coco.json", type=str)
    args = parser.parse_args()

    main(args)
