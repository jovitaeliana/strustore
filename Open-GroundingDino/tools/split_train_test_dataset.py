import os
import random
import json

def split_folders(root_base="../", folders=range(1, 6), train_ratio=0.7, seed=42):
    random.seed(seed)
    result = {"train": {}, "test": {}}

    for i in folders:
        folder_path = os.path.join(root_base, str(i))
        if not os.path.isdir(folder_path):
            print(f"Warning: folder {folder_path} does not exist. Skipping.")
            continue

        all_files = [f for f in os.listdir(folder_path)
                     if os.path.isfile(os.path.join(folder_path, f))]
        all_files.sort()  # optional: for consistency
        random.shuffle(all_files)

        n_train = int(len(all_files) * train_ratio)
        result["train"][str(i)] = all_files[:n_train]
        result["test"][str(i)] = all_files[n_train:]

        print(f"Folder {i}: {len(all_files)} files → {n_train} train / {len(all_files)-n_train} test")

    return result

if __name__ == "__main__":
    split_result = split_folders(root_base="../", folders=range(1, 6), train_ratio=0.7, seed=123)

    with open("../annotations/dataset_split.json", "w") as f:
        json.dump(split_result, f, indent=2)

    print("Split saved to dataset_split.json")
