import time
from pathlib import Path
from functools import partial
import json
import subprocess
import copy
import sys

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import schedulefree as sfoptim
from PIL import Image
from sklearn.decomposition import PCA


from data import (
    ImageNetImageDataset,
    RandomResizedCropAndInterpolation,
    center_crop_and_resize,
    fast_collate,
    PrefetchLoader,
    random_augment,
)
from dinov2 import vit_small, vit_large, DINOHead, TeacherStudent, DINOLoss


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
    patch_size=14,
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
    device = torch.device(device)

    ######## DATA ########
    train_ds = ImageNetImageDataset(
        path_to_imagenet, transform=RandomResizedCropAndInterpolation(image_size)
    )
    val_ds = ImageNetImageDataset(
        path_to_imagenet, transform=partial(center_crop_and_resize, size=image_size)
    )
    train_dl = PrefetchLoader(
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=fast_collate,
        ),
        device=device,
    )
    ####### MODEL ########
    teacher_backbone = vit_large(patch_size=patch_size, num_register_tokens=4)
    student_backbone = vit_large(
        patch_size=patch_size,
        num_register_tokens=4,
        drop_path_rate=0.4,
        drop_path_uniform=True,
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
            # img_batch is (B, H, W, 3)
            img_batch = img_batch.permute(0, 3, 1, 2)  # (B 3 H W)
            img_batch = random_augment(img_batch, vis=False)
            img_batch = img_batch / 255 - 0.5  # normalize [-0.5, 0.5]

            # now we need to create the masks that will leave vs visible patches for the student and vt for the teacher, and the teacher sees all those of the student.
            masks_student, masks_teacher = create_random_masks(
                B=img_batch.shape[0],
                L=(image_size // patch_size) ** 2,
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
