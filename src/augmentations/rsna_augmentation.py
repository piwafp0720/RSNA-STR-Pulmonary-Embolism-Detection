import albumentations as A
import cv2


class RSNAAugmentation:
    def __init__(self, size=512, mode='train'):
        assert mode in ['train', 'val', 'test']

        if mode == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    p=0.5,
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=10,
                    interpolation=1,
                    border_mode=4,
                ),
                A.OneOf([
                    A.Cutout(
                        p=1.0,
                        num_holes=6,
                        max_h_size=32,
                        max_w_size=32,
                    ),
                    A.GridDropout(
                        p=1.0,
                        ratio=0.5,
                        unit_size_min=64,
                        unit_size_max=128,
                        random_offset=True
                    ),
                ], p=0.5),
                A.Resize(size, size, p=1.0),
                ])
        elif mode == 'val':
            self.transform = A.Compose(
                [A.Resize(size, size, p=1.0),
                ])
        elif mode == 'test':
            self.transform = A.Compose(
                [A.Resize(size, size, p=1.0),
                ])

    def __call__(self, **kwargs):

        augmented = self.transform(**kwargs)
        img = augmented['image']

        return img

