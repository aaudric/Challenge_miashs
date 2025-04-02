import os
import random
from PIL import Image
from pathlib import Path

n = '4'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3}) 


# === Config ===
import os
import random
from PIL import Image
from pathlib import Path

# === Config ===
DATA_DIR = "../data/"
SAVE_DIR = "background_patches/"
NB_PATCHES_PER_IMAGE = 3
MIN_WIDTH, MAX_WIDTH = 200, 1200
MIN_HEIGHT, MAX_HEIGHT = 200, 1200

os.makedirs(SAVE_DIR, exist_ok=True)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0
    return interArea / float(boxAArea + boxBArea - interArea)

def read_bboxes_and_project(txt_path, img_width, img_height):
    bboxes = []
    project_name = None
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            if project_name is None:
                project_name = parts[1] + " " + parts[2]
            cx, cy, w, h = map(float, parts[-4:])
            abs_cx = cx * img_width
            abs_cy = cy * img_height
            abs_w = w * img_width
            abs_h = h * img_height
            bbox = (
                abs_cx - abs_w / 2,
                abs_cy - abs_h / 2,
                abs_cx + abs_w / 2,
                abs_cy + abs_h / 2
            )
            bboxes.append(bbox)
    return bboxes, project_name

def generate_background_patches(image_path, bboxes, project_name, img_id):
    img = Image.open(image_path).convert("RGB")
    patches = []
    tries = 0
    while len(patches) < NB_PATCHES_PER_IMAGE and tries < 100:
        patch_w = random.randint(MIN_WIDTH, MAX_WIDTH)
        patch_h = random.randint(MIN_HEIGHT, MAX_HEIGHT)

        if patch_w > img.width or patch_h > img.height:
            tries += 1
            continue

        x = random.randint(0, img.width - patch_w)
        y = random.randint(0, img.height - patch_h)
        patch_box = (x, y, x + patch_w, y + patch_h)

        if all(compute_iou(patch_box, bbox) < 0.05 for bbox in bboxes):
            patch = img.crop(patch_box)
            patch_name = f"{img_id}_bg_{len(patches)}.jpg"
            txt_name = f"{img_id}_bg_{len(patches)}.txt"
            patch.save(os.path.join(SAVE_DIR, patch_name))
            with open(os.path.join(SAVE_DIR, txt_name), 'w') as f:
                f.write(f"8_8_8_8 {project_name}\n")
            patches.append(patch)
        tries += 1

# === Boucle principale
data_dir = Path(DATA_DIR)
for image_path in data_dir.glob("*.jpg"):
    img_id = image_path.stem
    txt_path = data_dir / f"{img_id}.txt"
    if not txt_path.exists():
        continue
    img = Image.open(image_path)
    bboxes, project = read_bboxes_and_project(txt_path, img.width, img.height)
    if not bboxes:
        continue
    generate_background_patches(image_path, bboxes, project, img_id)