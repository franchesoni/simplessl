
from pathlib import Path
from functools import partial
import json
import subprocess
import copy
import sys

from simplejpeg import decode_jpeg
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import torch


from config import IMAGENET_PATH
from vit import vit_small


## DATA
class ImageNetImageDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = Path(path)
        if Path("image_paths_cache.npy").exists():
            self.image_paths = np.load("image_paths_cache.npy", allow_pickle=True)
        else:
            print(f"Loading image paths from {self.path}...")
            self.image_paths = np.array(sorted(self.path.glob('*.JPEG')))
            np.save("image_paths_cache.npy", self.image_paths)
        print('Done loading image paths.')
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = decode_jpeg(
            self.image_paths[idx].read_bytes(), fastdct=True, fastupsample=True
        )
        except Exception as e:
            print(f"Error reading {self.image_paths[idx]}: {e}")
            print("Returning a black image instead.")
            return np.zeros((224, 224, 3), dtype=np.uint8)
        img = self.transform(img)
        return img

# similar to timm/data/transforms.py
class RandomResizedCropAndInterpolation:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        assert scale[1] <= 1.0, "scale should be less than or equal to 1"
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.log_ratio = (np.log(ratio[0]), np.log(ratio[1]))

    def __call__(self, img):
        H, W= img.shape[:2]
        target_area = H * W * np.random.uniform(*self.scale)
        aspect_ratio = np.exp(np.random.uniform(*self.log_ratio))
        h = min(int(np.sqrt(target_area / aspect_ratio)), H)
        w = min(int(np.sqrt(target_area * aspect_ratio)), W)
        row = np.random.randint(0, H - h) if h != H else 0
        col = np.random.randint(0, W - w) if w != W else 0
        img = img[row:row+h, col:col+w]
        img = cv2.resize(img, (self.size, self.size))
        return img

def center_crop_and_resize(img, size=224):
    H, W = img.shape[:2]
    S = min(H, W)
    excess_h, excess_w = H - S, W - S
    img = img[excess_h//2:excess_h//2+S, excess_w//2:excess_w//2+S]
    img = cv2.resize(img, (size, size))
    return img


## MODEL




## TRAINING
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_current_git_commit():
    try:
        # Run the git command to get the current commit hash
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        # Decode from bytes to a string
        return commit_hash.decode("utf-8")
    except subprocess.CalledProcessError:
        # Handle the case where the command fails (e.g., not a Git repository)
        print("An error occurred while trying to retrieve the git commit hash.")
        return None



def main(batch_size=32, image_size=224, seed=0, tag="", dev=False):
    ####### SET UP (logging) ########
    seed_everything(seed)
    hparams = copy.deepcopy(locals())  # these are roughly the args passed to main
    command = " ".join(sys.argv)
    if not dev:
        logger = SummaryWriter(comment=tag)
        img_logdir = Path(logger.get_logdir()) / "images"
        img_logdir.mkdir(exist_ok=True, parents=True)
        with open(Path(logger.get_logdir()) / "hparams.txt", "w") as f:
            json.dump({"command": command, "git_commit": get_current_git_commit(), "script_file": Path(__file__), **hparams}, f)
    ######## DATA ########
    train_ds = ImageNetImageDataset(IMAGENET_PATH, transform=RandomResizedCropAndInterpolation(image_size))
    val_ds = ImageNetImageDataset(IMAGENET_PATH, transform=partial(center_crop_and_resize, size=image_size))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    for i in range(10):
        img_train = train_ds[i]
        img_test = val_ds[i]
        model = vit_small(patch_size=14, num_register_tokens=4)#, interpolate_antialias=True, interpolate_offset=0.0)
        breakpoint()



if __name__ == "__main__":
    from fire import Fire
    Fire(main)