from functools import lru_cache, partial
from pathlib import Path

import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

from dinov2code import vit_small, vit_large
from trainer import train, seed_everything, pct_norm



def mix_tokens(tokens, indices=None, seed=None, effective_batch_size=None):
    """
    Takes a tokens tensor of shape (B,L,D) and for each l in L, it will shuffle across the batch dimension.
    It's supposed to be reproducible given same shape and seed.
    It returns the indices used to shuffle the tokens.
    effective_batch_size is the number of batch elements mixed together. If None, it's the same as the batch size, i.e. all elements in the batch are mixed.
    """
    # we need to return a new batch tensor that has the same number of elements B but with the patch tokens mixed. The first 5 tokens should be always the same
    B, L, D = tokens.shape
    if effective_batch_size is None:
        effective_batch_size = B
    else:
        assert (B % effective_batch_size == 0), "Batch size must be divisible by effective batch size"
    if indices is None:
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        noise = torch.rand(B, L, device=tokens.device)
        for offset in range(B // effective_batch_size):
            noise[offset * effective_batch_size : (offset + 1) * effective_batch_size] += offset 
        indices = torch.argsort(
            noise, dim=0
        )  # (B, 256), random indices that reorder patches for a given L
    elif indices.shape[1] != L:
        raise ValueError("Indices should have the same number of tokens as the tokens tensor")
    return (
        torch.gather(tokens, dim=0, index=indices.unsqueeze(2).expand(B, L, D)),
        indices,
    )


def unmix_tokens(mixed_tokens, indices):
    B, L, D = mixed_tokens.shape
    assert L == indices.shape[1], "Indices should have the same number of tokens as the tokens tensor but got {} and {}".format(L, indices.shape[1])

    # Create a tensor of indices to sort back to the original order
    unshuffle_indices = torch.argsort(indices, dim=0)

    # Use these indices to gather the original tokens
    original_tokens = torch.gather(
        mixed_tokens, dim=0, index=unshuffle_indices.unsqueeze(2).expand(B, L, D)
    )

    return original_tokens


def image_to_patches(image, patch_size):
    B, C, H, W = image.shape
    assert (
        H % patch_size == 0 and W % patch_size == 0
    ), "patch_size must divide both H and W"

    # Unfold the image into patches
    patches = F.unfold(image, kernel_size=patch_size, stride=patch_size)

    # Reshape to (B, C * patch_size * patch_size, L)
    patches = patches.view(B, C * patch_size * patch_size, -1)

    # Transpose to get (B, L, D)
    patches = patches.transpose(1, 2)

    return patches


def patches_to_image(patches, patch_size, H, W):
    B, L, D = patches.shape
    assert D == 3 * patch_size * patch_size, f"We expect rgb_tokens with D = {3 * patch_size * patch_size} but found D = {D}"

    # Transpose back to (B, D, L)
    patches = patches.transpose(1, 2)

    # Use fold to reconstruct the image
    image = F.fold(
        patches, output_size=(H, W), kernel_size=patch_size, stride=patch_size
    )  # (B, C, H, W)

    return image


def raster_reconstruction(tokens):
    out = tokens.clone()
    B, L, D = out.shape
    side = int(L ** (1 / 2))
    out = out.reshape(B, side, side, D)
    indices = np.empty((B, side, side), dtype=int)
    for row in range(side):
        for col in range(side):
            if row == 0 and col == 0:
                continue
            if col > 0:
                neighbor = out[:, row, col - 1]
            else:
                neighbor = out[:, row - 1, col]
            cur = out[:, row, col]  # (B, D)
            indices_cur = align_second_with_first(neighbor, cur)  # (B, D), (B,)
            out[:, row, col] = out[indices_cur, row, col]
            indices[:, row, col] = indices_cur
    out = out.reshape(B, L, D)
    indices = torch.argsort(torch.from_numpy(indices).reshape(B, L), dim=0).to(
        tokens.device
    )  # the argsort is for further use with the unmix
    return out, indices


def align_second_with_first(x1, x2):
    assert x1.shape == x2.shape
    assert len(x1.shape) == 2
    dist = torch.cdist(x1, x2).cpu().numpy()
    _, col_indices = linear_sum_assignment(dist)
    return col_indices


@lru_cache(maxsize=None)
def block_diag(num_blocks, block_size):
    """
    Create a block diagonal matrix with ones in square blocks along the diagonal.

    Args:
    num_blocks (int): Number of blocks along the diagonal.
    block_size (int): Size of each square block.

    Returns:
    torch.Tensor: A 2D tensor representing the block diagonal matrix.
    """
    total_size = num_blocks * block_size

    # Create a range tensor
    r = torch.arange(total_size)

    # Create conditions for block diagonal
    condition1 = (r.unsqueeze(1) // block_size) == (r.unsqueeze(0) // block_size)
    condition2 = (r.unsqueeze(1) // block_size) == (r.unsqueeze(0) // block_size)

    # Combine conditions
    mask = condition1 & condition2

    return mask.float()


def loss(unmixed_tokens):
    # the loss is MSE(KTK - I) where K are the D, BL tokens and I is a block diagonal with ones for each batch
    # we first compute the term for the block diagonal and then compute the rest
    B, L, D = unmixed_tokens.shape
    unmixed_tokens = torch.nn.functional.normalize(unmixed_tokens, p=2, dim=-1)
    batched_cosine_similarity = torch.bmm(unmixed_tokens, unmixed_tokens.transpose(1, 2))
    loss = torch.sum(torch.abs(batched_cosine_similarity - 1)) - torch.sum(torch.abs(batched_cosine_similarity)) 
    unmixed_tokens = unmixed_tokens.view(-1, D)
    loss += torch.sum(torch.abs(torch.matmul(unmixed_tokens, unmixed_tokens.T)))  # (BL, BL)
    return loss / ((B * L) ** 2)


def validate(
    val_ds,
    model,
    device,
    global_step,
    img_logdir,
    writer,
    n_batches,
    batch_size,
    num_workers,
    seed=0,
    patch_size=14,
    image_size=224,
    n_show=16,
):
    # This eval function should look at the loss of many batches of images. In fact it should look at the loss of each mixed image in each batch, in order to see if the quantity of evaluated images is enough to get a good performance estimate. Make sure it's reproducible and that different method look at the same images. This is similar to what we did in main.py.
    generator = torch.Generator()
    generator.manual_seed(seed)
    dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    losses = []
    for i, img_batch in enumerate(dl):
        if i == n_batches:
            print("Done evaluating", end="\r")
            break
        img_batch = (img_batch.to(device) / 255 - 0.5).permute(0, 3, 1, 2)

        rgb_tokens = image_to_patches(img_batch, patch_size=patch_size)
        mixed_rgb_tokens, indices = mix_tokens(rgb_tokens)  # these indices work for the patch tokens only, indices is (B, L)
        mixed_img_batch = patches_to_image(mixed_rgb_tokens, patch_size, img_batch.shape[2], img_batch.shape[3])  # (B, 3, H, W)
        mixed_tokens = model.forward_features(mixed_img_batch)["x_norm_patchtokens"]  # (B, L, D) 
        unmixed_tokens = unmix_tokens(mixed_tokens, indices)
        losses.append(loss(unmixed_tokens).item())

        _, rasterindices = raster_reconstruction(
           mixed_tokens 
        )  # gives the indices given by raster similarity

        print(f"Mean loss at batch {i+1}/{n_batches}:", np.mean(losses), end="\r")



        if i < n_show:
            # for visualization, save the mixed images
            mixed_imgs = torch.concatenate(list(mixed_img_batch), dim=-1)
            Image.fromarray(
                (((mixed_imgs + 0.5) * 255).permute(1, 2, 0).cpu().numpy()).astype(
                    np.uint8
                )
            ).save(Path(img_logdir) / f"{i}_mixed_step_{global_step}.png")
            # for visualization, save the reconstructed images
            unmixed_rgb = unmix_tokens(mixed_rgb_tokens, rasterindices)
            reconstructed_imgs = patches_to_image(
                unmixed_rgb, patch_size, image_size, image_size
            )
            reconstructed_imgs = torch.concatenate(list(reconstructed_imgs), dim=-1)
            Image.fromarray(
                (
                    ((reconstructed_imgs + 0.5) * 255).permute(1, 2, 0).cpu().numpy()
                ).astype(np.uint8)
            ).save(Path(img_logdir) / f"{i}_reconstructed_step_{global_step}.png")
            # do pca
            pca = PCA(n_components=3)
            B, L, D = unmixed_tokens.shape
            pca_features = pca.fit_transform(
                unmixed_tokens.reshape(B * L, D).cpu().numpy()
            ).reshape(B, L, 3)
            pca_features = pct_norm(pca_features).reshape(B, int(L ** (1 / 2)), int(L ** (1 / 2)), 3)
            pca_features = np.concatenate(list(pca_features), axis=1)
            raw_imgs = torch.concatenate(list(((img_batch+0.5)*255).permute(0, 2, 3, 1)), dim=1)
            Image.fromarray((pca_features*255).astype(np.uint8)).save(img_logdir / f"{i}_pca_step_{global_step}.png")
            Image.fromarray(raw_imgs.cpu().numpy().astype(np.uint8)).save(img_logdir / f"{i}_raw_step_{global_step}.png")

    # now compute the loss
    if writer:
        writer.add_histogram(
            tag="val/loss", values=np.array(losses), global_step=global_step
        )
        writer.add_scalar("val/mean_loss", np.mean(losses), global_step)
        # save network
        torch.save(model.state_dict, Path(writer.get_logdir()) / f"last_validated_model.pth")
    print("Mean validation loss:", np.mean(losses))
    return np.mean(losses)


class RGBModel(torch.nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        # add a dummy parameter to trick the optimizer
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def prepare_tokens_with_masks(self, img_batch, masks=None):
        return image_to_patches(img_batch, patch_size=self.patch_size)

    def forward_tokens(self, tokens):
        return {"x_norm_patchtokens": tokens}

    def forward_features(self, x):
        return self.forward_tokens(self.prepare_tokens_with_masks(x))


class FrozenDinoLinear(torch.nn.Module):
    def __init__(self, model, dino_weights=False, frozen=False, linear=False):
        super().__init__()
        if dino_weights:
            # we make explicit that the models architectures are the same by loading the weights instead of the model
            model.load_state_dict(
                torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14_reg"
                ).state_dict()
            )
        else:
            raise ValueError("Why are you using this class if you're not using DINO?")
        self.backbone = model
        if linear:
            self.proj = torch.nn.Linear(384, 384)
        else:
            self.proj = torch.nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def prepare_tokens_with_masks(self, img_batch, masks=None):
        return self.backbone.prepare_tokens_with_masks(img_batch, masks)

    def forward_tokens(self, tokens):
        return {
            "x_norm_patchtokens": self.proj(
                self.backbone.forward_tokens(tokens)["x_norm_patchtokens"]
            )
        }

    def forward_features(self, x):
        return self.forward_tokens(self.prepare_tokens_with_masks(x))


def init_model_and_loss(patch_size, model_name="vits"):
    assert model_name in ["vits", "vitl", "rgb", "dinov2regs", "frozendinov2regs_linear"]
    if model_name in ["vits", "dinov2regs", "frozendinov2regs_linear"]:
        model = vit_small(
            patch_size,
            init_values=1.0,
            img_size=518,
            block_chunks=0,
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
        )
        if "dinov2regs" in model_name:
            assert patch_size == 14, "Dinov2regs only works with patch size 14"
            model = FrozenDinoLinear(
                model,
                dino_weights=True,
                frozen="frozen" in model_name,
                linear="linear" in model_name,
            )
    elif model_name == "rgb":
        model = RGBModel(patch_size)
    elif model_name == "vitl":
        model = vit_large(
            patch_size,
            init_values=1.0,
            img_size=518,
            block_chunks=0,
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            # drop_path_rate=0.2,
            # drop_path_uniform=True,
        )
    else:
        raise NotImplementedError

    model.train()
    return model, loss


def dummy_train_step(*args, **kwargs):
    return {"total_loss": torch.tensor([0])}

def visualize_image_batch(img_batch, n_images, tag):
    B, C, H, W = img_batch.shape
    assert img_batch.min() > -0.51 and img_batch.max() < 0.51
    images = ((img_batch.permute(0, 2, 3, 1).cpu().numpy() + 0.5) * 255).astype(np.uint8)
    for b in range(min(B, n_images)):
        Image.fromarray(images[b]).save(f"tmp/{tag}_{b}.png")
    


def train_step(
    img_batch,
    model,
    loss_fn,
    optimizer,
    scheduler,
    effective_batch_size,
    patch_size,
):
    assert (
        -0.51 <= img_batch.min() and img_batch.max() <= 0.51
    ), f"Images should be normalized to [-0.5, 0.5] but are in [{img_batch.min()}, {img_batch.max()}]"
    assert img_batch.shape[1] == 3, f"Images should have 3 channels but have shape {img_batch.shape}"
    # img_batch is (B, 3, H, W)
    # get the images, mix them, forward them, unmix the tokens, compute loss
    rgb_tokens = image_to_patches(img_batch, patch_size=patch_size)
    mixed_rgb_tokens, indices = mix_tokens(rgb_tokens, effective_batch_size=effective_batch_size)  # these indices work for the patch tokens only, indices is (B, L)
    mixed_img_batch = patches_to_image(mixed_rgb_tokens, patch_size, img_batch.shape[2], img_batch.shape[3])  # (B, 3, H, W)
    mixed_tokens = model.forward_features(mixed_img_batch)["x_norm_patchtokens"]  # (B, L, D) 
    unmixed_tokens = unmix_tokens(mixed_tokens, indices)
    total_loss = loss_fn(unmixed_tokens)
    # optimizer step
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {"total_loss": total_loss}


def main(
    model_name="rgb",
    n_val_batches=1000,
    evaluate=False,
    imagenet_path="/export/home/data/imagenet",
    steps=10000,
    val_every=1000,
    batch_size=4,
    effective_batch_size=4,
    val_batch_size=4,
    num_workers=0,
    image_size=224,
    patch_size=14,
    lr=0.001,
    weight_decay=0.0001,
    beta=0.9,
    seed=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dev=False,
    sing=False,
    tag="",
):

    model_loss_init_fn = partial(
        init_model_and_loss, patch_size=patch_size, model_name=model_name
        # note that we don't init the model with the given image size, which conditions the size of the positional embedding
    )
    if evaluate:
        train_step_fn = dummy_train_step
        steps = 1
    else:
        train_step_fn = partial(train_step, effective_batch_size=effective_batch_size, patch_size=patch_size)
    validate_fn = partial(
        validate,
        n_batches=n_val_batches,
        num_workers=num_workers,
        batch_size=val_batch_size,
        seed=seed,
        patch_size=patch_size,
        image_size=image_size,
    )
    train(
        model_loss_init_fn=model_loss_init_fn,
        train_step_fn=train_step_fn,
        validate_fn=validate_fn,
        path_to_imagenet=imagenet_path,
        steps=steps,
        val_every=val_every,
        lr=lr,
        beta=beta,
        weight_decay=weight_decay,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        device=device,
        seed=seed,
        dev=dev,
        tag=tag,
        sing=sing,
    )

    # img_batch = next(iter(dl))  # img_batch is (B, H, W, 3)
    # [
    #     Image.fromarray(img_batch[i].numpy()).save(f"tmp/{i}_original.png")
    #     for i in range(batch_size)
    # ]
    # img_batch = img_batch.permute(0, 3, 1, 2).float()  # (B 3 H W)
    # img_batch = random_augment(img_batch, vis=False)
    # [
    #     Image.fromarray((img_batch[i].permute(1, 2, 0).numpy()).astype(np.uint8)).save(
    #         f"tmp/{i}_augmented.png"
    #     )
    #     for i in range(batch_size)
    # ]
    # # img_batch = img_batch / 255 - 0.5  # normalize [-0.5, 0.5]

    # # model = vit_small(patch_size=14, num_register_tokens=4)
    # # tokens = model.prepare_tokens_with_masks(img_batch, masks=None)
    # tokens = image_to_patches(img_batch, patch_size=14)
    # images = patches_to_image(tokens, patch_size=14, H=224, W=224)
    # [
    #     Image.fromarray((images[i].permute(1, 2, 0).numpy()).astype(np.uint8)).save(
    #         f"tmp/{i}_reconstructed_correct.png"
    #     )
    #     for i in range(batch_size)
    # ]
    # tokens, indices = mix_tokens(tokens)
    # new_tokens1, rasterindices = raster_reconstruction(tokens)
    # unmixed = unmix_tokens(tokens, indices)
    # unmixedraster = unmix_tokens(tokens, rasterindices)

    # images = patches_to_image(tokens, patch_size=14, H=224, W=224)
    # [
    #     Image.fromarray((images[i].permute(1, 2, 0).numpy()).astype(np.uint8)).save(
    #         f"tmp/{i}_mixed.png"
    #     )
    #     for i in range(batch_size)
    # ]
    # images1 = patches_to_image(new_tokens1, patch_size=14, H=224, W=224)
    # [
    #     Image.fromarray((images1[i].permute(1, 2, 0).numpy()).astype(np.uint8)).save(
    #         f"tmp/{i}_reconstructed1.png"
    #     )
    #     for i in range(batch_size)
    # ]
    # unmixed_imgs = patches_to_image(unmixed, patch_size=14, H=224, W=224)
    # [
    #     Image.fromarray(
    #         (unmixed_imgs[i].permute(1, 2, 0).numpy()).astype(np.uint8)
    #     ).save(f"tmp/{i}_unmixed.png")
    #     for i in range(batch_size)
    # ]
    # raster_unmixed_imgs = patches_to_image(unmixedraster, patch_size=14, H=224, W=224)
    # [
    #     Image.fromarray(
    #         (raster_unmixed_imgs[i].permute(1, 2, 0).numpy()).astype(np.uint8)
    #     ).save(f"tmp/{i}_raster_unmixed.png")
    #     for i in range(batch_size)
    # ]


"""
We have a loss, we have functions to mix and unmix, we have a function to get a dummy reconstruction based on the embeddings, that we can use to unmix. 
Therefore we have what we need for quantitative and qualitative evaluation of different networks / methods.
What we want to see next is whether this qualitative and quantitative evaluations correspond to each other.
With that aim, we will try, in order of performance:
    - random vit as embedding
    - rgb as embedding
    - dinov2reg as embedding

What we expect to see is that the random vit will not be nice, the rgb a little nicer and dinov2reg the nicest, both qualitative and quantitatively.
If the dinov2reg doesn't work well, I'm sure a linear layer trained on top will work better. This should be our loss performance baseline.

next:
- [DONE] Create an evaluation function for an embedder. This eval function should look at the loss of many batches of images. In fact it should look at the loss of each mixed image in each batch, in order to see if the quantity of evaluated images is enough to get a good performance estimate. Make sure it's reproducible and that different method look at the same images. This is similar to what we did in main.py.
- [DONE] Evaluate the random vit and register results.
- [DONE] Evaluate the rgb and register results.
- [DONE] Evaluate dinov2reg and register results.
- [DONE] Train a linear layer on top of dinov2reg and save the checkpoint.
- [DONE] Evaluate the lienar layer on top of dinov2reg and register results.
- [DONE] Train a vitS network from scratch. <- the loss is okish, not much better than dino+linear
- [DONE] Train a vitL network from scratch. <- didn't work, loss jumped up, need sing
- [DONE] Use SING optimizer to see if we get rid of the jumps in the loss
- [DONE] Visualize PCA
- Make patch size flexible
- Visualize retrieval 
- Retrain a vitL network from scratch.
- Train a vitL network with patches of 7 and bigger batch size

"""
if __name__ == "__main__":
    from fire import Fire

    Fire(main)

# RESULTS:

# python patcher.py --n_val_batches=256 --tag="eval_patch_rgb" --evaluate --steps=1 --model_name=rgb
# Mean validation loss: 0.49649881210643810643874/s|  GPU mem 0.06 GB

# python patcher.py --n_val_batches=256 --tag="eval_patch_vits" --evaluate --steps=1 --model_name=vits
# Mean validation loss: 0.34940459206700325700325/s|  GPU mem 0.24 GB

# python patcher.py --n_val_batches=256 --tag="eval_patch_dinov2regs" --evaluate --steps=1 --model_name=dinov2regs
# Mean validation loss: 0.23852409579558298558298/s|  GPU mem 0.24 GB

# be careful with validation, changing the batch_size yield different results 
# on the following configuration (frozen dino + linear) we see that 1. the linear layer doesn't degrade performance at the begginging and 2. it can optimize the loss quite a bit
# python patcher.py --n_val_batches=256 --tag="frozendino_linear" --steps=10000 --val_every=2000 --batch_size=32 --num_workers=64 --model_name=frozendinov2regs_linear --device="cuda:1"
# Mean validation loss: 0.24118619802175092175092step/s|  GPU mem 1.97 GB  (one gradient descent step)
# Mean validation loss: 0.17905898584285754285757+01 step/s|  GPU mem 1.97 GB  (2k gradient descent steps)
# Mean validation loss: 0.17991738574346527346527+01 step/s|  GPU mem 1.97 GB  (8k gradient descent steps)

# the following vits worked ok:
# python patcher.py --n_val_batches=256 --tag="vits" --steps=10000 --val_every=2000 --batch_size=32 --num_workers=64 --model_name=vits --device="cuda:1" --dev

# the following vitl had a jump in the loss and never recovered
#  python patcher.py --n_val_batches=256 --tag="vitl" --steps=20001 --val_every=2000 --batch_size=32 --num_workers=64 --model_name=vitl --device="cuda:1"


# python patcher.py --imagenet_path="/HomeToo/data/stage/image_datasets/train_blurred" --model_name='vits' --batch_size=96 --num_workers=48 --n_val_batches=100 --steps=40000 --val_every=2000 --tag="vits"