import time
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
import schedulefree as sfoptim
from PIL import Image


from config import IMAGENET_PATH
from dinov2 import vit_small, DINOHead, TeacherStudent, DINOLoss


## DATA
class ImageNetImageDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = Path(path)
        print(f"Loading image paths from {self.path}...", end="\r")
        if Path("image_paths_cache.npy").exists():
            self.image_paths = np.load("image_paths_cache.npy", allow_pickle=True)
        else:
            self.image_paths = np.array(sorted(self.path.glob("*.JPEG")))
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
            except Exception as e:  # if PIL also doesn't work, report the image and return a black or a white image.
                print(f"Error reading with simplejpeg and PIL {self.image_paths[idx]}: {e}. The image is probably corrupted, returning a plain image.")
                return np.ones((224, 224, 3), dtype=np.uint8) * 255 * (idx % 2),  # return a white or black image
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


def create_random_masks(B, L, V1, V2, device):
    """
    Creates two masks tensors of shape (B, L) that have exactly V positive entries per batch element.
    """
    # Create a tensor of indices
    indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

    # Create random permutations for each batch
    permutations = torch.argsort(torch.rand(B, L, device=device), dim=1)

    # Use the permutations to shuffle the indices
    shuffled_indices = torch.gather(indices, 1, permutations)

    # Create the mask by comparing with M
    masks1 = shuffled_indices < V1
    masks2 = shuffled_indices < V2
    return masks1, masks2


cfg = dict(  # default dino config
    dino=dict(head_n_prototypes=65536),
)


def main(
    path_to_imagenet="/export/home/data/imagenet",  # the expected data structure is IMAGENET_PATH/image_name.JPEG where image_name is the name of the image in the train set. We use no val or test set.
    steps=1000,
    patches_student=8,
    patches_teacher=32,
    teacher_momentum=0.994,
    lr=0.01,
    beta=0.9,
    weight_decay=0.0,
    batch_size=256,
    num_workers=48,
    image_size=224,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=0,
    tag="",
    dev=False,
):
    ####### SET UP (logging) ########
    seed_everything(seed)
    hparams = copy.deepcopy(locals())  # these are roughly the args passed to main
    command = " ".join(sys.argv)
    if not dev:
        writer = SummaryWriter(comment=tag)
        img_logdir = Path(writer.get_logdir()) / "images"
        img_logdir.mkdir(exist_ok=True, parents=True)
        with open(Path(writer.get_logdir()) / "hparams.txt", "w") as f:
            json.dump(
                {
                    "command": command,
                    "git_commit": get_current_git_commit(),
                    "script_file": str(Path(__file__)),
                    **hparams,
                },
                f,
            )

    ######## DATA ########
    train_ds = ImageNetImageDataset(
        path_to_imagenet, transform=RandomResizedCropAndInterpolation(image_size)
    )
    val_ds = ImageNetImageDataset(
        path_to_imagenet, transform=partial(center_crop_and_resize, size=image_size)
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    ####### MODEL ########
    teacher_backbone = vit_small(patch_size=14, num_register_tokens=4)
    student_backbone = vit_small(patch_size=14, num_register_tokens=4)
    teacher_head = DINOHead(
        in_dim=teacher_backbone.embed_dim, out_dim=cfg["dino"]["head_n_prototypes"]
    )
    student_head = DINOHead(
        in_dim=student_backbone.embed_dim, out_dim=cfg["dino"]["head_n_prototypes"]
    )
    models = TeacherStudent(
        teacher_backbone, student_backbone, teacher_head, student_head
    ).to(device)
    dino_loss = DINOLoss(out_dim=cfg["dino"]["head_n_prototypes"]).to(device)

    ######## OPTIMIZER ########
    optimizer = sfoptim.AdamWScheduleFree(
        models.parameters(),
        lr=lr,
        warmup_steps=steps // 20,
        betas=(beta, 0.999),
        weight_decay=weight_decay,
    )
    models.train()
    models.teacher.eval()  # teacher shouldn't be trained
    optimizer.train()

    ####### LOOP #########
    global_step = 0
    st = time.time()
    while global_step < steps:
        for imgs in train_dl:
            # imgs is (B, 3, H, W)
            imgs = (imgs.to(device, non_blocking=True) / 255 - 0.5).permute(
                0, 3, 1, 2
            )  # go from uint8 numpy to float32 torch tensor B C H W
            # now we need to create the masks that will leave vs visible patches for the student and vt for the teacher, and the teacher sees all those of the student.
            masks_student, masks_teacher = create_random_masks(
                B=imgs.shape[0],
                L=256,
                V1=patches_student,
                V2=patches_teacher,
                device=device,
            )
            # let us use simple dino loss
            with torch.no_grad():
                out_teacher = models.teacher.backbone(imgs, masks_teacher)
                teacher_cls_tokens = out_teacher["x_norm_clstoken"]
                teacher_cls_tokens_after_head = models.teacher.head(teacher_cls_tokens)
                teacher_dino_softmaxed_centered = dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=0.05
                )
                dino_loss.update_center(teacher_cls_tokens_after_head)
            out_student = student_backbone(imgs, masks_student)
            student_cls_tokens = out_student["x_norm_clstoken"]
            student_cls_tokens_after_head = models.student.head(student_cls_tokens)
            dino_global_loss = dino_loss(
                student_output_list=[student_cls_tokens_after_head],
                teacher_out_softmaxed_centered_list=[teacher_dino_softmaxed_centered],
            )
            total_loss = dino_global_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            models.update_teacher(teacher_momentum)

            writer.add_scalar("train/total_loss", total_loss.item(), global_step)
            print(
                f"{global_step}/{steps}={(global_step / steps * 100):.2f}%| "
                f"Loss {total_loss.item():.3e}| "
                f"Speed {(global_step + 1) / (time.time() - st):.3e} step/s| ",
                f"GPU mem {(torch.cuda.memory_allocated(device) + torch.cuda.memory_reserved(device)) / 1e9:.2f} GB",
                end="\r",
            )

            global_step += 1
            if global_step >= steps:
                break

    # log and checkpoint
    with open(writer.get_logdir() / "done.txt", "w") as f:
        f.write("done")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
