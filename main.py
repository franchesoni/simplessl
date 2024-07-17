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
from sklearn.decomposition import PCA
import torchvision.transforms.v2 as transforms


from dinov2 import vit_small, vit_large, DINOHead, TeacherStudent, DINOLoss


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


jitter, grayscale, blur, solarize = (
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.Grayscale(),
    transforms.GaussianBlur(kernel_size=9),
    transforms.RandomSolarize(threshold=128),
)


def pseudo_random_augment(img_batch):
    """
    Random augmentations following DINOv2. Flip with 0.5 prob, color jittering with 0.8 prob, grayscale with 0.2 prob, blur with 0.5, solarize with 0.2.
    Because we have a batch, we favor pseudo-random augmentations. We will apply augmentations to the proportion of the images given by the probabilities.
    """
    B = img_batch.shape[0]
    # 1. we flip the first half of the batch, as the batch is random it's a random flip
    img_batch[: B // 2] = torch.flip(img_batch[: B // 2], dims=[3])  # last half
    # 2. apply color jitter to 80% of the images, half of them are flipped. Ignore which images are flipped
    img_batch[: (8 * B) // 10] = jitter(img_batch[: (8 * B) // 10])  # first 80%
    # 3. apply grayscale to 20% of the images, Ignore which images are flipped or jittered
    img_batch[: B // 5] = grayscale(img_batch[: B // 5])
    # 4. apply gaussian blur to half of the images, all jittered, all flipped, few grayscale (10%-60%)
    img_batch[B // 10 : (6 * B) // 10] = blur(img_batch[B // 10 : (6 * B) // 10])
    # 5. apply solarize to the images that are not color jittered
    img_batch[-B // 5 :] = solarize(img_batch[-B // 5 :])
    return img_batch


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


def pct_norm(x, p=1):
    newmin, newmax = np.percentile(x, p), np.percentile(x, 100 - p)
    return np.clip((x - newmin) / (newmax - newmin), 0, 1)


def validate(
    val_ds, models, device, global_step, img_logdir, n_imgs=8, search_among=128
):
    assert search_among % n_imgs == 0, "search_among must be divisible by n_imgs"
    generator = torch.Generator()  # always sample the same images
    generator.manual_seed(0)
    dl = DataLoader(
        val_ds, batch_size=n_imgs, shuffle=True, num_workers=0, generator=generator
    )
    ######### QUALITATIVE #########
    library_feats, library_imgs = [], []
    for i, raw_imgs in enumerate(dl):
        imgs = (raw_imgs.to(device) / 255 - 0.5).permute(0, 3, 1, 2)
        out = models.teacher.backbone(imgs)
        if i == 0:
            # First do a pca of the embeddings for each patch and save hte color images
            pca = PCA(n_components=3)
            patch_features = out["x_norm_patchtokens"]
            B, L, D = patch_features.shape
            rgb_features = pca.fit_transform(
                patch_features.reshape(B * L, D).cpu().numpy()
            ).reshape(B, L, 3)
            rgb_features = pct_norm(rgb_features).reshape(
                B, int(L ** (1 / 2)), int(L ** (1 / 2)), 3
            )
            for i, raw_img in enumerate(raw_imgs):
                Image.fromarray(raw_img.numpy()).save(
                    str(img_logdir / f"step_{global_step}_{i}_color.jpg")
                )
                Image.fromarray((rgb_features[i] * 255).astype(np.uint8)).save(
                    str(img_logdir / f"step_{global_step}_{i}_pca.jpg")
                )
            # we keep the first batch of images to compare with the rest in retrieval
            reference_feats = out["x_norm_clstoken"]  # (B, D)
            reference_imgs = raw_imgs  # (B, H, W, 3)
        else:
            library_feats.append(out["x_norm_clstoken"])  # (B, D)
            library_imgs.append(raw_imgs)
        if i == search_among // n_imgs:
            break
    library_feats = torch.cat(library_feats, dim=0)  # (search_among, D)
    library_imgs = torch.cat(library_imgs, dim=0)  # (search_among, H, W, 3)
    # compute cosine similarity
    reference_feats, library_feats = reference_feats / reference_feats.norm(
        dim=-1, keepdim=True
    ), library_feats / library_feats.norm(dim=-1, keepdim=True)
    sim = torch.einsum("bd,cd->bc", reference_feats, library_feats)  # (B, search_among)
    retrieval_indices = sim.argmax(dim=1).cpu()  # (B,)
    for i in range(n_imgs):
        joined_img = torch.cat(
            [reference_imgs[i], library_imgs[retrieval_indices[i]]], dim=1
        )
        Image.fromarray(joined_img.cpu().numpy()).save(
            str(img_logdir / f"step_{global_step}_{i}_retrieval.jpg")
        )


def main(
    path_to_imagenet="/export/home/data/imagenet",  # the expected data structure is IMAGENET_PATH/image_name.JPEG where image_name is the name of the image in the train set. We use no val or test set.
    steps=10000,
    val_every=1000,
    patches_student=8,
    patches_teacher=32,
    out_dim=65536,  # default DINOv2 out_dim
    teacher_momentum=0.997,
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
    if dev:
        writer = None
        img_logdir = Path("tmp")
        img_logdir.mkdir(exist_ok=True, parents=True)
    else:
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
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    ####### MODEL ########
    teacher_backbone = vit_large(patch_size=14, num_register_tokens=4)
    student_backbone = vit_large(
        patch_size=14, num_register_tokens=4, drop_path_rate=0.4, drop_path_uniform=True
    )
    teacher_head = DINOHead(
        in_dim=teacher_backbone.embed_dim,
        out_dim=out_dim,
    )
    student_head = DINOHead(
        in_dim=student_backbone.embed_dim,
        out_dim=out_dim,
    )
    models = TeacherStudent(
        teacher_backbone, student_backbone, teacher_head, student_head
    ).to(device)
    dino_loss = DINOLoss(out_dim=out_dim).to(device)

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
        for img_batch in train_dl:
            # imgs is (B, 3, H, W)
            img_batch = img_batch.to(device, non_blocking=True).permute(
                0, 3, 1, 2
            )  # go from uint8 numpy to float32 torch tensor B C H W
            img_batch = pseudo_random_augment(img_batch)
            img_batch = img_batch / 255 - 0.5  # normalize [-0.5, 0.5]

            # now we need to create the masks that will leave vs visible patches for the student and vt for the teacher, and the teacher sees all those of the student.
            masks_student, masks_teacher = create_random_masks(
                B=img_batch.shape[0],
                L=256,
                V1=patches_student,
                V2=patches_teacher,
                device=device,
            )
            # let us use simple dino loss
            with torch.no_grad():
                out_teacher = models.teacher.backbone(img_batch, masks_teacher)
                teacher_cls_tokens = out_teacher["x_norm_clstoken"]
                teacher_cls_tokens_after_head = models.teacher.head(teacher_cls_tokens)
                teacher_dino_softmaxed_centered = dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=0.05
                )
                dino_loss.update_center(teacher_cls_tokens_after_head)
            out_student = student_backbone(img_batch, masks_student)
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

            if writer:
                writer.add_scalar("train/total_loss", total_loss.item(), global_step)
            print(
                f"{global_step}/{steps}={(global_step / steps * 100):.2f}%| "
                f"Loss {total_loss.item():.3e}| "
                f"Speed {(global_step + 1) / (time.time() - st):.3e} step/s| ",
                f"GPU mem {(torch.cuda.memory_allocated(device) + torch.cuda.memory_reserved(device)) / 1e9:.2f} GB",
                end="\r",
            )

            if global_step % val_every == 0:
                print("Validating qualitatively...", end="\r")
                with torch.no_grad():
                    validate(
                        val_ds,
                        models,
                        device,
                        global_step,
                        img_logdir,
                    )
                    # compute RankMe
                    print("Computing RankMe...", end="\r")
                    singular_values = torch.linalg.svdvals(
                        models.teacher.head.last_layer.weight  # (out_dim, in_dim)
                    )
                    pks = (
                        singular_values / torch.linalg.norm(singular_values, ord=1)
                        + 1e-7
                    )
                    rankme = torch.exp(-torch.sum(pks * torch.log(pks)))
                    if not dev:
                        writer.add_scalar("train/rankme", rankme.item(), global_step)
                        # save teacher
                        teacher_state_dict = models.teacher.state_dict()
                        torch.save(
                            teacher_state_dict,
                            Path(writer.get_logdir()) / "last_validated_teacher.pth",
                        )

            global_step += 1
            if global_step >= steps:
                break

    # log and checkpoint
    if not dev:
        with open(Path(writer.get_logdir()) / "done.txt", "w") as f:
            f.write("done")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
