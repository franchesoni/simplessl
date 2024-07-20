from pathlib import Path
import json
from functools import partial
import time
import sys
import subprocess
import copy

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import schedulefree as sfoptim

from data import ImageNetImageDataset, RandomResizedCropAndInterpolation, center_crop_and_resize, fast_collate, PrefetchLoader, random_augment
from sing import SING

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


def train(
    model_loss_init_fn,  # should return the inintialized model(s) and loss function/class
    train_step_fn,  # should return a losses dict and update the models
    validate_fn,
    path_to_imagenet="/export/home/data/imagenet",  # the expected data structure is IMAGENET_PATH/image_name.JPEG where image_name is the name of the image in the train set. We use no val or test set.
    steps=10000,
    val_every=1000,
    patch_size=14,
    lr=0.01,
    beta=0.9,
    weight_decay=0.0001,
    batch_size=256,
    num_workers=48,
    image_size=224,
    device="cuda",
    seed=0,
    tag="",
    dev=False,
    sing=False,
):
    ####### SET UP (logging) ########
    if (not dev) and ((steps - 1)  % val_every != 0):
        raise ValueError("validate_every does not evaluate the last step")
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
                    **{k: str(v) if not (isinstance(v, int) or isinstance(v, bool)) else v for k, v in hparams.items()},
                },
                f,
            )
    if not torch.cuda.is_available() and "cuda" in device:
        device = "cpu"
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
    model, loss_fn = model_loss_init_fn()
    model = model.to(device)
    loss_fn = loss_fn.to(device) if hasattr(loss_fn, "to") else loss_fn


    ######## OPTIMIZER ########
    if sing:
        optimizer = SING(model.parameters(), lr=lr/10, weight_decay=weight_decay, betas=(beta, 0.999))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=steps + 1,
            pct_start=0.05,
            final_div_factor=1e12,
        )
    else:
        optimizer = sfoptim.AdamWScheduleFree(
            model.parameters(),
            lr=lr,
            warmup_steps=steps // 20,
            betas=(beta, 0.999),
            weight_decay=weight_decay,
        )
        optimizer.train()
        scheduler = None

    ####### LOOP #########
    global_step = 0
    st = time.time()
    while global_step < steps:
        for img_batch in train_dl:
            # img_batch is (B, H, W, 3)
            img_batch = img_batch.permute(0, 3, 1, 2)  # (B 3 H W)
            img_batch = random_augment(img_batch, vis=False)
            img_batch = img_batch / 255 - 0.5  # normalize [-0.5, 0.5]

            losses = train_step_fn(img_batch, model, loss_fn, optimizer, scheduler)

            if writer:
                for loss_name in losses:
                    writer.add_scalar(f"train/{loss_name}", losses[loss_name].item(), global_step)
            print(
                f"{global_step}/{steps}={(global_step / steps * 100):.2f}%| "
                f"Loss {losses["total_loss"].item():.3e}| "
                f"Speed {(global_step + 1) / (time.time() - st):.3e} step/s| ",
                f"GPU mem {(torch.cuda.memory_allocated(device) + torch.cuda.memory_reserved(device)) / 1e9:.2f} GB",
                end="\r",
            )

            if global_step % val_every == 0:
                print("Validating...", end="\r")
                with torch.no_grad():
                    validate_fn(
                        val_ds,
                        model,
                        device,
                        global_step,
                        img_logdir,
                        writer,
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

    Fire(train)
