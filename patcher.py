import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np
from functools import lru_cache
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

from data import (
    ImageNetImageDataset,
    RandomResizedCropAndInterpolation,
    fast_collate,
    random_augment,
)
from dinov2 import vit_small


def mix_tokens(tokens, indices=None):
    # we need to return a new batch tensor that has the same number of elements B but with the patch tokens mixed. The first 5 tokens should be always the same
    # assert torch.allclose(tokens[0, :5], tokens[-1, :5]), "The first 5 tokens should be the same for all images in the batch"
    B, L, D = tokens.shape
    if indices is None:
        indices = torch.argsort(
            torch.rand(B, L), dim=0
        )  # random indices that reorder patches for a given L
    return (
        torch.gather(tokens, dim=0, index=indices.unsqueeze(2).expand(B, L, D)),
        indices,
    )


def unmix_tokens(mixed_tokens, indices):
    B, L, D = mixed_tokens.shape

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
    C = D // (patch_size * patch_size)

    # Transpose back to (B, D, L)
    patches = patches.transpose(1, 2)

    # Reshape to match F.fold input shape
    patches = patches.view(B, C * patch_size * patch_size, L)

    # Use fold to reconstruct the image
    image = F.fold(
        patches, output_size=(H, W), kernel_size=patch_size, stride=patch_size
    )

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
    indices = torch.argsort(torch.from_numpy(indices).reshape(B, L), dim=0)  # the argsort is for further use with the unmix
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
    B, L, D = unmixed_tokens.shape
    unmixed_tokens = unmixed_tokens.view(B * L, D)
    unmixed_tokens = unmixed_tokens / torch.norm(
        unmixed_tokens, dim=1, keepdim=True
    )  # normalize (BL, D)
    cosine_similarity = torch.matmul(unmixed_tokens, unmixed_tokens.T)  # (BL, BL)
    mask = block_diag(B, L).to(unmixed_tokens.device)  # (BL, BL)
    return torch.mean((cosine_similarity - mask) ** 2)


batch_size = 4
num_workers = 0
ds = ImageNetImageDataset(
    "/export/home/data/imagenet", transform=RandomResizedCropAndInterpolation(224)
)
dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=True,
    collate_fn=fast_collate,
)

img_batch = next(iter(dl))  # img_batch is (B, H, W, 3)
[
    Image.fromarray(img_batch[i].numpy()).save(f"tmp/{i}_original.png")
    for i in range(batch_size)
]
img_batch = img_batch.permute(0, 3, 1, 2).float()  # (B 3 H W)
img_batch = random_augment(img_batch, vis=False)
[
    Image.fromarray((img_batch[i].permute(1, 2, 0).numpy()).astype(np.uint8)).save(
        f"tmp/{i}_augmented.png"
    )
    for i in range(batch_size)
]
# img_batch = img_batch / 255 - 0.5  # normalize [-0.5, 0.5]

# model = vit_small(patch_size=14, num_register_tokens=4)
# tokens = model.prepare_tokens_with_masks(img_batch, masks=None)
tokens = image_to_patches(img_batch, patch_size=14)
images = patches_to_image(tokens, patch_size=14, H=224, W=224)
[
    Image.fromarray((images[i].permute(1, 2, 0).numpy()).astype(np.uint8)).save(
        f"tmp/{i}_reconstructed_correct.png"
    )
    for i in range(batch_size)
]
tokens, indices = mix_tokens(tokens)
new_tokens1, rasterindices = raster_reconstruction(tokens)
unmixed = unmix_tokens(tokens, indices)
unmixedraster = unmix_tokens(tokens, rasterindices)

images = patches_to_image(tokens, patch_size=14, H=224, W=224)
[
    Image.fromarray((images[i].permute(1, 2, 0).numpy()).astype(np.uint8)).save(
        f"tmp/{i}_mixed.png"
    )
    for i in range(batch_size)
]
images1 = patches_to_image(new_tokens1, patch_size=14, H=224, W=224)
[
    Image.fromarray((images1[i].permute(1, 2, 0).numpy()).astype(np.uint8)).save(
        f"tmp/{i}_reconstructed1.png"
    )
    for i in range(batch_size)
]
unmixed_imgs = patches_to_image(unmixed, patch_size=14, H=224, W=224)
[
    Image.fromarray((unmixed_imgs[i].permute(1, 2, 0).numpy()).astype(np.uint8)).save(
        f"tmp/{i}_unmixed.png"
    )
    for i in range(batch_size)
]
raster_unmixed_imgs = patches_to_image(unmixedraster, patch_size=14, H=224, W=224)
[
    Image.fromarray((raster_unmixed_imgs[i].permute(1, 2, 0).numpy()).astype(np.uint8)).save(
        f"tmp/{i}_raster_unmixed.png"
    )
    for i in range(batch_size)
]


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
- Create an evaluation function for an embedder. This eval function should look at the loss of many batches of images. In fact it should look at the loss of each mixed image in each batch, in order to see if the quantity of evaluated images is enough to get a good performance estimate. Make sure it's reproducible and that different method look at the same images. This is similar to what we did in main.py.
- Evaluate the random vit and register results.
- Evaluate the rgb and register results.
- Evaluate dinov2reg and register results.
- Train a linear layer on top of dinov2reg and save the checkpoint.
- Evaluate the lienar layer on top of dinov2reg and register results.
- Train a network from scratch.
"""