import os
import pandas as pd
from torchvision.datasets import Flowers102
from torchvision import transforms
from PIL import Image

# 1. Download Flowers102 (use a split)
dataset = Flowers102(root="data", split="train", download=True)

# 2. Create output folders
save_dir = "data/Flowers102"
img_dir = os.path.join(save_dir, "images")
os.makedirs(img_dir, exist_ok=True)

# 3. Transform for resizing (optional, for CLIP)
resize_transform = transforms.Resize((224,224))

# 4. Save subset of images + captions
records = []
N = min(1020, len(dataset))  # prevent IndexError

FLOWERS102_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "ball moss",
    "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily", "common tulip", "wild rose",
    "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cactus flower", "water lily", "rose",
    "thorn apple", "morning glory", "passion flower", "lotus", "toad lily",
    "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
    "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
    "bee balm", "ball cactus", "foxglove beardtongue", "petunia", "wild pansy",
    "primrose", "sunflower (variant)", "violet", "cyclamen persicum", "azalea",
    "anthurium andraeanum", "carnation", "giant white arum lily", "common dandelion",
    "hibiscus rosa-sinensis", "mexican sunflower", "osteospermum", "daffodil",
    "wild daffodil", "snowdrop", "lily of the valley", "fritillary", "iris",
    "windflower", "corn poppy", "poppy anemone", "gazania", "gerbera", "african daisy",
    "passionvine", "garden phlox", "balloon flower", "buttercup", "chrysanthemum",
    "calendula", "daisy", "marguerite daisy", "mallow hibiscus", "evening primrose",
    "camomile", "scarlet flax", "black-eyed susan", "oxeye daisy", "gaillardia",
    "arctic poppy", "rose of sharon", "gladiolus"
]


for i in range(N):
    img, label = dataset[i]   # img is PIL.Image, label is int
    img = resize_transform(img)  # resize for CLIP

    # Convert label index → class name
    class_name = FLOWERS102_CLASSES[label]

    # Save image
    fname = f"img_{i:04d}.jpg"
    path = os.path.join(img_dir, fname)
    img.save(path, "JPEG")

    # Generate simple caption
    caption = f"A photo of a {class_name}"
    records.append({"image_path": path, "caption": caption, "label": class_name})

# 5. Save captions CSV
df = pd.DataFrame(records)
csv_path = os.path.join(save_dir, "captions.csv")
df.to_csv(csv_path, index=False)

print(f"Saved {N} images to {img_dir}")
print(f"Captions saved to {csv_path}") 
