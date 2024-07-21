from pathlib import Path
from functools import partial
from contextlib import suppress

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from simplejpeg import decode_jpeg
import torchvision.transforms.v2 as transforms
import cv2


## DATA
class ImageNetImageDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = Path(path)
        print(f"Loading image paths from {self.path}...", end="\r")
        if Path("image_paths_cache.npy").exists():
            self.image_paths = np.load("image_paths_cache.npy", allow_pickle=True)
        else:
            self.image_paths = []
            for ext in ['jpg', 'jpeg', 'JPG', 'JPEG']:
                self.image_paths += list(self.path.glob(f"**/*.{ext}")) 
                if len(self.image_paths) > 100000:  # assume we found them all
                    break
            self.image_paths = np.array(sorted(self.image_paths))
            np.save("image_paths_cache.npy", self.image_paths)
        print("Done loading image paths.", end="\r")
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:  # simplejpeg is fastest
            img = decode_jpeg(
                self.image_paths[idx].read_bytes(), fastdct=True, fastupsample=True
            )
        except ValueError as e:
            print(f"Error reading with simplejpeg", end="\r")
            print("Trying with PIL...", end="\r")
            try:  # if it doesn't work use PIL, decently fast
                img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
                print("Read with PIL!", end="\r")
            except (
                Exception
            ) as e:  # if PIL also doesn't work, report the image and return a black or a white image.
                print(
                    f"Error reading with simplejpeg and PIL {self.image_paths[idx]}: {e}. The image is probably corrupted, returning a plain image."
                )
                return (
                    np.ones((224, 224, 3), dtype=np.uint8) * 255 * (idx % 2),
                )  # return a white or black image
        img = self.transform(img)
        return img


# similar to timm/data/transforms.py
class RandomResizedCropAndInterpolation:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        assert scale[1] <= 1.0, "scale should be less than or equal to 1"
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.log_ratio = (np.log(ratio[0]), np.log(ratio[1]))

    def __call__(self, img):
        H, W = img.shape[:2]
        target_area = H * W * np.random.uniform(*self.scale)
        aspect_ratio = np.exp(np.random.uniform(*self.log_ratio))
        h = min(int(np.sqrt(target_area / aspect_ratio)), H)
        w = min(int(np.sqrt(target_area * aspect_ratio)), W)
        row = np.random.randint(0, H - h) if h != H else 0
        col = np.random.randint(0, W - w) if w != W else 0
        img = img[row : row + h, col : col + w]
        img = cv2.resize(img, (self.size, self.size))
        return img


def center_crop_and_resize(img, size=224):
    H, W = img.shape[:2]
    S = min(H, W)
    excess_h, excess_w = H - S, W - S
    img = img[excess_h // 2 : excess_h // 2 + S, excess_w // 2 : excess_w // 2 + S]
    img = cv2.resize(img, (size, size))
    return img


grayscale, blur, solarize = (
    transforms.Grayscale(),
    transforms.GaussianBlur(kernel_size=9),
    transforms.RandomSolarize(p=1., threshold=0.5),
)


def random_augment(img_batch, vis=False):
    """
    Random augmentations following DINOv2. Flip with 0.5 prob, color jittering with 0.8 prob, grayscale with 0.2 prob, blur with 0.5, solarize with 0.2.
    Because we have a batch, we favor pseudo-random augmentations. We will apply augmentations to the proportion of the images given by the probabilities.
    """
    assert img_batch.ndim == 4, "img_batch should be (B, C, H, W)"
    assert img_batch.max() > 1. and img_batch.max() < 256. and img_batch.min() >= 0, "img_batch should be in [0, 255.]"
    if vis:
        for i in range(10):
            Image.fromarray(
                (img_batch[i].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            ).save(f"tmp/{i}_raw.png")
    B = img_batch.shape[0]
    # 1. we flip the first half of the batch, as the batch is random it's a random flip
    img_batch[: B // 2] = torch.flip(img_batch[: B // 2], dims=[3])  # last half

    # The following commented lines apply the same color jitter, we can do more diverse, ignoring hue
    # # 2. apply color jitter to 80% of the images, half of them are flipped. Ignore which images are flipped
    # img_batch[: (8 * B) // 10] = jitter(img_batch[: (8 * B) // 10])  # first 80%
    gray_batch = img_batch.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    mean_gray_batch = gray_batch.mean(dim=[2, 3], keepdim=True)  # (B, 1, 1, 1)
    bright_f, contrast_f, saturation_f = (
        torch.rand(3, (8 * B) // 10, 1, 1, 1, device=img_batch.device) * 2
    )
    img_batch[: (8 * B) // 10] = torch.clip(
        (
            img_batch[: (8 * B) // 10] * (bright_f + contrast_f + saturation_f)
            + mean_gray_batch[: (8 * B) // 10] * (1 - contrast_f)
            + gray_batch[: (8 * B) // 10] * (1 - saturation_f)
        )
        // 3,
        0,
        255,
    )

    # 3. apply grayscale to 20% of the images, Ignore which images are flipped or jittered
    img_batch[: B // 5] = gray_batch[: B // 5].repeat(1, 3, 1, 1)
    # 4. apply gaussian blur to half of the images, all jittered, all flipped, few grayscale (10%-60%)
    img_batch[B // 10 : (6 * B) // 10] = blur(img_batch[B // 10 : (6 * B) // 10])
    # 5. apply solarize to the images that are not color jittered
    img_batch[-B // 5 :] = solarize(img_batch[-B // 5 :] / 255) * 255
    if vis:
        for i in range(10):
            Image.fromarray(
                (img_batch[i].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            ).save(f"tmp/{i}_transformed.png")

    return img_batch


def fast_collate(batch):
    return torch.from_numpy(np.array(batch))


class PrefetchLoader:

    def __init__(
        self,
        loader,
        device,
        img_dtype=torch.float32,
    ):

        self.loader = loader
        self.device = device
        self.img_dtype = img_dtype
        self.is_cuda = torch.cuda.is_available() and device.type == "cuda"

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress

        for next_input in self.loader:

            with stream_context():
                next_input = next_input.to(device=self.device, non_blocking=True)
                next_input = next_input.to(self.img_dtype)

            if not first:
                yield input
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)

            input = next_input

        yield input

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
