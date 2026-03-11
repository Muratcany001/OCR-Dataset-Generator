import albumentations as A


class DataAugmenter:
    def __init__(self, augmentations=None):
        """
        augmentations: dict
            Örn: {"blur": {"blur_limit": 7, "p": 1.0},
                  "skew": {"shear_x": (-15, 15), "shear_y": (-10, 10), "rotate": (-5,5), "scale": (0.9,1.1), "p":0.7}}
        """
        self.transforms = augmentations if augmentations else {}

    def get_operations(self):
        operations = {}
        for name, params in self.transforms.items():
            if name == "blur":
                operations[name] = A.Blur(**params)
            elif name == "skew":
                operations[name] = A.Affine(
                    shear={
                        "x": params.get("shear_x", (0, 0)),
                        "y": params.get("shear_y", (0, 0)),
                    },
                    rotate=params.get("rotate", (0, 0)),
                    scale=params.get("scale", (1, 1)),
                    keep_ratio=True,
                    p=params.get("p", 1.0),
                )
            elif name == "brightness":
                operations[name] = A.RandomBrightnessContrast(p=params.get("p", 0.5))
        return operations
