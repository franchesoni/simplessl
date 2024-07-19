from pathlib import Path
from functools import partial

from torch.utils.data import DataLoader
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA


from dinov2code import vit_small, vit_large, DINOHead, TeacherStudent, DINOLoss
from trainer import train


def init_model_and_loss(patch_size, out_dim):
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
    )
    dino_loss = DINOLoss(out_dim=out_dim)
    models.train()
    models.teacher.eval()
    return models, dino_loss


def train_step(
    img_batch,
    models,
    loss_fn,
    optimizer,
    image_size,
    patch_size,
    device,
    patches_student,
    patches_teacher,
    teacher_momentum,
):
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
        teacher_dino_softmaxed_centered = loss_fn.softmax_center_teacher(
            teacher_cls_tokens_after_head, teacher_temp=0.05
        )
        loss_fn.update_center(teacher_cls_tokens_after_head)
    out_student = models.student.backbone(img_batch, masks_student)
    student_cls_tokens = out_student["x_norm_clstoken"]
    student_cls_tokens_after_head = models.student.head(student_cls_tokens)
    dino_global_loss = loss_fn(
        student_output_list=[student_cls_tokens_after_head],
        teacher_out_softmaxed_centered_list=[teacher_dino_softmaxed_centered],
    )
    total_loss = dino_global_loss
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()
    models.update_teacher(teacher_momentum)
    return {"total_loss": total_loss}


## TRAINING


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
    val_ds, models, device, global_step, img_logdir, writer, n_imgs=8, search_among=128
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

    # compute RankMe
    print("Computing RankMe...", end="\r")
    singular_values = torch.linalg.svdvals(
        models.teacher.head.last_layer.weight  # (out_dim, in_dim)
    )
    pks = singular_values / torch.linalg.norm(singular_values, ord=1) + 1e-7
    rankme = torch.exp(-torch.sum(pks * torch.log(pks)))
    if writer:
        writer.add_scalar("train/rankme", rankme.item(), global_step)
        # save teacher
        teacher_state_dict = models.teacher.state_dict()
        torch.save(
            teacher_state_dict,
            Path(writer.get_logdir()) / "last_validated_teacher.pth",
        )


def main(
    out_dim=65536,  # this is the default DINOv2 out_dim
    patches_student=8,
    patches_teacher=32,
    teacher_momentum=0.997,
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
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=0,
    tag="",
    dev=False,
):
    model_loss_init_fn = partial(
        init_model_and_loss, patch_size=patch_size, out_dim=out_dim
    )
    train_step_fn = partial(
        train_step,
        image_size=image_size,
        patch_size=patch_size,
        device=device,
        patches_student=patches_student,
        patches_teacher=patches_teacher,
        teacher_momentum=teacher_momentum,
    )
    validate_fn = partial(validate, n_imgs=8, search_among=128)
    train(
        model_loss_init_fn,
        train_step_fn,
        validate_fn,
        path_to_imagenet,
        steps,
        val_every,
        patch_size,
        lr,
        beta,
        weight_decay,
        batch_size,
        num_workers,
        image_size,
        device,
        seed,
        tag,
        dev,
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
