import numpy as np

SEX_FEATURES = {
    "Female": 1,
    "Male": 2,
    "Unknown": 3
}

LOCALIZATION_FEATURES = {
    "abdomen": 4,
    "acral": 5,
    "back": 6,
    "chest": 7,
    "ear": 8,
    "face": 9,
    "foot": 10,
    "genital": 11,
    "hand": 12,
    "lower extremity": 13,
    "neck": 14,
    "scalp": 15,
    "trunk": 16,
    "unknown": 17,
    "upper extremity": 18
}


def encode_metadata(age, sex, localization):
    """
    Creates metadata vector with shape (1, 19)

    Feature order:
    0  age
    1  sex_female
    2  sex_male
    3  sex_unknown
    4  localization_abdomen
    5  localization_acral
    6  localization_back
    7  localization_chest
    8  localization_ear
    9  localization_face
    10 localization_foot
    11 localization_genital
    12 localization_hand
    13 localization_lower extremity
    14 localization_neck
    15 localization_scalp
    16 localization_trunk
    17 localization_unknown
    18 localization_upper extremity
    """

    metadata = np.zeros((1, 19), dtype=np.float32)

    # Age normalization
    metadata[0, 0] = age / 100.0

    # Sex one-hot encoding
    if sex in SEX_FEATURES:
        metadata[0, SEX_FEATURES[sex]] = 1.0

    # Localization one-hot encoding
    if localization in LOCALIZATION_FEATURES:
        metadata[0, LOCALIZATION_FEATURES[localization]] = 1.0

    return metadata


if __name__ == "__main__":
    test = encode_metadata(
        age=45,
        sex="Male",
        localization="back"
    )

    print(test)
    print("Shape:", test.shape)