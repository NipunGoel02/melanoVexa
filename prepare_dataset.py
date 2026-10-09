import os
import shutil
import pandas as pd

CSV = os.path.expanduser(
    r"~/Downloads/ISIC_2019_Training_GroundTruth.csv"
)

IMAGE_DIR = r"raw_data\ISIC_2019_Training_Input"

SEED = 42

df = pd.read_csv(CSV)

# Remove unknowns
df = df[df["UNK"] == 0].copy()

# Convert one-hot labels to class names
classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

df["label"] = df[classes].idxmax(axis=1)

# ============================================================
# MULTICLASS
# MEL: 2713 train + 678 val
# BCC: 1993 train + 498 val
# SCC:  376 train +  94 val
# ============================================================

multi_counts = {
    "MEL": (2713, 678),
    "BCC": (1993, 498),
    "SCC": (376, 94),
}

# ============================================================
# BINARY
# NV:     7725 train + 1931 val
# Others: 7470 train + 1865 val
# ============================================================

binary_counts = {
    "NV": (7725, 1931),
    "OTHERS": (7470, 1865),
}


def copy_image(image_id, destination):
    src = os.path.join(IMAGE_DIR, image_id + ".jpg")
    dst = os.path.join(destination, image_id + ".jpg")

    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing image: {src}")

    shutil.copy2(src, dst)


def prepare_multiclass():
    print("\n========== MULTICLASS ==========")

    for cls, (train_n, val_n) in multi_counts.items():

        data = df[df["label"] == cls].sample(
            n=train_n + val_n,
            random_state=SEED
        )

        train = data.iloc[:train_n]
        val = data.iloc[train_n:train_n + val_n]

        train_dir = os.path.join(
            "data_mult", "train", cls
        )
        val_dir = os.path.join(
            "data_mult", "val", cls
        )

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for image_id in train["image"]:
            copy_image(image_id, train_dir)

        for image_id in val["image"]:
            copy_image(image_id, val_dir)

        print(
            f"{cls}: train={len(train)}, val={len(val)}"
        )


def prepare_binary():
    print("\n========== BINARY ==========")

    # NV
    nv = df[df["label"] == "NV"].sample(
        n=7725 + 1931,
        random_state=SEED
    )

    nv_train = nv.iloc[:7725]
    nv_val = nv.iloc[7725:]

    # Others = every non-NV class
    others = df[df["label"] != "NV"].sample(
        n=7470 + 1865,
        random_state=SEED
    )

    others_train = others.iloc[:7470]
    others_val = others.iloc[7470:]

    splits = [
        ("NV", nv_train, "train"),
        ("NV", nv_val, "val"),
        ("OTHERS", others_train, "train"),
        ("OTHERS", others_val, "val"),
    ]

    for cls, data, split in splits:

        destination = os.path.join(
            "data", split, cls
        )

        os.makedirs(destination, exist_ok=True)

        for image_id in data["image"]:
            copy_image(image_id, destination)

        print(
            f"{cls}: {split}={len(data)}"
        )


if __name__ == "__main__":

    print("Images in CSV:", len(df))

    prepare_multiclass()
    prepare_binary()

    print("\n================================")
    print("DATASET PREPARATION COMPLETE")
    print("================================")