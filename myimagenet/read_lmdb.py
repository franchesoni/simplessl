import torch
torch.multiprocessing.set_start_method("fork", force=True)
import time
import lmdb
from torch.utils.data import Dataset
import tqdm
from simplejpeg import decode_jpeg


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,  # Optimize for sequential access
            meminit=False,     # Avoid unnecessary memory initialization
            map_size=0,
        )
        with self.env.begin() as txn:
            self.length = txn.stat()['entries'] // 2  # Assuming each entry has an image and a label
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            img_key = str(idx).encode("utf-8")
            lbl_key = f"L{idx}".encode("utf-8")
            img_data = txn.get(img_key)
            label = int(txn.get(lbl_key).decode())
        # print(f"Image key: {img_key}, Label key: {lbl_key}, Label value: {label}, Image size: {len(img_data)}", end='\r')
        img = decode_jpeg(img_data, fastdct=True, fastupsample=True)
        # if self.transform:
            # img = self.transform(img)

        return img



if __name__ == "__main__":
    # lmdb_path = "/home/franchesoni/data/imagenet1k/train.lmdb"
    lmdb_path = "/dev/shm/imagenet/train.lmdb"

    # Example: without transforms
    dataset = LMDBDataset(lmdb_path)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        # num_workers=0,
        # num_workers=24,
        # persistent_workers=True,
        collate_fn=lambda x: x,
    )
    print(len(dataset), len(dataloader))

    for batch in tqdm.tqdm(dataloader):
        pass

# # Verify
# import lmdb

# env = lmdb.open(lmdb_path, readonly=True)
# with env.begin() as txn:
#     cursor = txn.cursor()
#     for key, value in cursor:
#         print(key, len(value))
#         if b"_label" in key:
#             print(f"Label key: {key}, Label value: {value.decode('utf-8')}")
# env.close()


# -------------------------------------

# import lmdb
# import torch
# from torch.utils.data import Dataset
# import tqdm
# from simplejpeg import decode_jpeg

# # Global variable for shared memory
# def load_lmdb_into_memory(lmdb_path):
#     """Loads LMDB contents into a shared dictionary."""
#     env = lmdb.open(lmdb_path, readonly=True, lock=False)
#     lmdb_cache = {}
#     with env.begin() as txn:
#         st = time.time()
#         i = 0
#         while True:
#             img_key = str(i).encode("utf-8")
#             img_data = txn.get(img_key)
#             if img_data is None:
#                 break
#             lbl_key = f"L{i}".encode("utf-8")
#             label = txn.get(lbl_key)
#             lmdb_cache[i] = (img_data, int(label.decode()))
#             i += 1
#             if i % 10000 == 0:
#                 print('iter', i, i / (time.time() - st), 'it/s      ', end='\r')

#     env.close()
#     return lmdb_cache

# class LMDBDataset(Dataset):
#     def __init__(self, lmdb_path, transform=None):
#         self.lmdb_cache =load_lmdb_into_memory(lmdb_path)
#         self.keys = list(self.lmdb_cache.keys())
#         self.transform = transform

#     def __len__(self):
#         return len(self.keys)

#     def __getitem__(self, idx):
#         img_data, label = self.lmdb_cache[self.keys[idx]]
#         img = decode_jpeg(img_data, fastdct=True, fastupsample=True)
#         if self.transform:
#             img = self.transform(img)
#         return img, label


# lmdb_path = "/home/franchesoni/data/imagenet1k/train.lmdb"

# # Example: without transforms
# # cache = load_lmdb_into_memory(lmdb_path)
# dataset = LMDBDataset(lmdb_path)

# print('ds len', len(dataset))
# dl1 = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=1024,
#     shuffle=False,
#     num_workers=0,
#     collate_fn=lambda x: x,
# )
# for batch in tqdm.tqdm(dl1):
#     pass
# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=1024,
#     shuffle=False,
#     num_workers=12,
#     collate_fn=lambda x: x,
# )



# for batch in tqdm.tqdm(dataloader):
#     pass

