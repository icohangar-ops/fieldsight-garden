"""Canonical class list for FieldSight Garden (PlantDoc, cropped classification release).

Each PlantDoc folder name is mapped to a clean "plant + condition" string. The string is
exactly what the model is trained to emit, so keep it short, lowercase and unambiguous.
"""

# PlantDoc folder name -> (canonical label, plant, condition, note)
FOLDER_TO_LABEL: dict[str, tuple[str, str, str, str]] = {
    "Apple Scab Leaf": ("apple scab", "apple", "scab", ""),
    "Apple leaf": ("apple healthy", "apple", "healthy", ""),
    "Apple rust leaf": ("apple rust", "apple", "rust", "Most likely cedar-apple rust."),
    "Bell_pepper leaf spot": ("bell pepper leaf spot", "bell pepper", "leaf spot",
                               "PlantVillage counterpart is bacterial spot; PlantDoc only says 'leaf spot'."),
    "Bell_pepper leaf": ("bell pepper healthy", "bell pepper", "healthy", ""),
    "Blueberry leaf": ("blueberry healthy", "blueberry", "healthy", ""),
    "Cherry leaf": ("cherry healthy", "cherry", "healthy", ""),
    "Corn Gray leaf spot": ("corn gray leaf spot", "corn", "gray leaf spot", ""),
    "Corn leaf blight": ("corn leaf blight", "corn", "leaf blight",
                          "PlantVillage counterpart is northern corn leaf blight."),
    "Corn rust leaf": ("corn rust", "corn", "rust", "PlantVillage counterpart is common rust."),
    "Peach leaf": ("peach healthy", "peach", "healthy", ""),
    "Potato leaf early blight": ("potato early blight", "potato", "early blight", ""),
    "Potato leaf late blight": ("potato late blight", "potato", "late blight", ""),
    "Raspberry leaf": ("raspberry healthy", "raspberry", "healthy", ""),
    "Soyabean leaf": ("soybean healthy", "soybean", "healthy", ""),
    "Squash Powdery mildew leaf": ("squash powdery mildew", "squash", "powdery mildew", ""),
    "Strawberry leaf": ("strawberry healthy", "strawberry", "healthy", ""),
    "Tomato Early blight leaf": ("tomato early blight", "tomato", "early blight", ""),
    "Tomato Septoria leaf spot": ("tomato septoria leaf spot", "tomato", "septoria leaf spot", ""),
    "Tomato leaf bacterial spot": ("tomato bacterial spot", "tomato", "bacterial spot", ""),
    "Tomato leaf late blight": ("tomato late blight", "tomato", "late blight", ""),
    "Tomato leaf mosaic virus": ("tomato mosaic virus", "tomato", "mosaic virus", ""),
    "Tomato leaf yellow virus": ("tomato yellow leaf curl virus", "tomato", "yellow leaf curl virus",
                                  "PlantDoc folder is 'Tomato leaf yellow virus'; mapped to TYLCV, the "
                                  "PlantVillage class it was modelled on (assumption)."),
    "Tomato leaf": ("tomato healthy", "tomato", "healthy", ""),
    "Tomato mold leaf": ("tomato leaf mold", "tomato", "leaf mold", ""),
    "grape leaf black rot": ("grape black rot", "grape", "black rot", ""),
    "grape leaf": ("grape healthy", "grape", "healthy", ""),
    # Dropped: only 2 train images and 0 test images in PlantDoc.
    "Tomato two spotted spider mites leaf": (None, "tomato", "two-spotted spider mites",
                                              "DROPPED: 2 images total."),
}

DROPPED_FOLDERS = {k for k, v in FOLDER_TO_LABEL.items() if v[0] is None}

CLASSES: list[str] = sorted({v[0] for v in FOLDER_TO_LABEL.values() if v[0] is not None})
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
