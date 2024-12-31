import os
import cv2
import lmdb
from pathlib import Path
from simplejpeg import decode_jpeg, encode_jpeg
import random
import io
import tqdm

def create_lmdb(data_dir, lmdb_path, longest_side=500):
    # Shuffle dataset
    print('Listing files...')
    data_dir = Path(data_dir)
    classes = sorted(os.listdir(data_dir))
    all_files = []
    for class_name in tqdm.tqdm(classes):
        class_path = data_dir / class_name 
        all_files.extend([(class_path / img, class_name) for img in os.listdir(class_path)])
    print(len(all_files))
    print('Shuffling...')
    random.seed(0)
    random.shuffle(all_files)
    
    print("Writing...")
    # Write to LMDB
    env = lmdb.open(lmdb_path, map_size=100 * 1024 ** 3)  # Adjust map_size as needed  (100GB)
    with env.begin(write=True) as txn:
        for idx, (img_path, class_name) in tqdm.tqdm(enumerate(all_files), total=len(all_files), smoothing=0.1):
            # read and resize
            img = decode_jpeg(img_path.read_bytes(), fastdct=True, fastupsample=True)
            height, width, _ = img.shape
            scale = longest_side / max(height, width)
            new_height, new_width = int(height * scale), int(width * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            img_data = encode_jpeg(img, quality=90, fastdct=True)
            # save
            key = str(idx).encode("utf-8")  # note that the real order will not be determined by idx but by str(idx) which is different
            label = classes.index(class_name)
            txn.put(key, img_data)
            txn.put(f"L{idx}".encode("utf-8"), str(label).encode("utf-8"))
        print("Closing...")
        print(f"Total: {len(all_files)}")
    env.close()
    print("Awesome!")

# Usage
import shutil
split = "train"
outpath = f"/home/franchesoni/data/imagenet1k/{split}.lmdb"
shutil.rmtree(outpath, ignore_errors=True)
create_lmdb(f"/home/franchesoni/data/imagenet1k/{split}", outpath)


