import os
import sys
import random
import numpy as np
import torch
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as T

sys.path.append(os.path.join(os.path.dirname(__file__), "EPIC/src"))

from models import create_backbone_model, create_modified_head
from matrix import create_matrix
from data import create_indexed_dataloader
from prototypes import (
    generate_prototypes,
    get_image_prototypes,
    get_visualized_prototypes,
    topk_active_channels,
    get_purity_fn,
)
import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "demo_config.yaml")


def load_config(path):
    """
    Load config from YAML, merging with defaults so missing keys are always
    available. If the file doesn't exist yet, write the defaults there so the
    user has a template to edit.
    """
    cfg_path = path or DEFAULT_CONFIG_PATH
 
    with open(cfg_path) as f:
        user_cfg = yaml.safe_load(f) or {}
 
    print(f"Loaded config from '{cfg_path}'")
    return user_cfg



cfg = load_config(DEFAULT_CONFIG_PATH )
MODEL_NAME       = cfg["model"]["name"]
NUM_CLASSES      = cfg["model"]["num_classes"]
BACKBONE_PATH    = cfg["model"]["backbone_path"]
MATRIX_TYPE      = cfg["matrix"]["type"]
MATRIX_PATH      = cfg["matrix"]["path"]
TRAIN_DATA_PATH  = cfg["data"]["train_path"]
BATCH_SIZE       = cfg["data"]["batch_size"]
NUM_WORKERS      = cfg["data"]["num_workers"]
TOP_K            = cfg["demo"]["top_k"]
NUM_PROTOTYPES   = cfg["demo"]["num_prototypes"]
PROTOTYPE_CACHE  = cfg["demo"]["prototype_cache"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


print("Loading backbone…")
base_model, feature_model, epic_transform, num_channels = create_backbone_model(
    MODEL_NAME,
    device=device,
    custom_weights_path=BACKBONE_PATH,
    num_classes=NUM_CLASSES,
)
base_model    = base_model.to(device).eval()
feature_model = feature_model.to(device).eval()

print("Loading disentanglement matrix…")
disentanglement_matrix = create_matrix(MATRIX_TYPE, num_channels, device)
disentanglement_matrix.load_state(MATRIX_PATH, map_location=device)
U = disentanglement_matrix()

classification_head = create_modified_head(base_model, MODEL_NAME, disentanglement_matrix)

print("Building training dataloader for prototype lookup…")
dataloader_train = create_indexed_dataloader(
    TRAIN_DATA_PATH,
    epic_transform,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    shuffle=False,
)

print("Generating prototypes (runs once, may take a minute)…")
purity_fn = get_purity_fn("argmax")
if PROTOTYPE_CACHE and os.path.exists(PROTOTYPE_CACHE):
    print(f"Loading cached prototypes from '{PROTOTYPE_CACHE}'...")
    positive_prototypes = torch.load(PROTOTYPE_CACHE, map_location=device)
else:
    positive_prototypes = generate_prototypes(
        feature_model,
        dataloader_train,
        num_channels,
        NUM_PROTOTYPES,
        device,
        U,
    )
    if PROTOTYPE_CACHE:
        torch.save(positive_prototypes, PROTOTYPE_CACHE)
        print(f"Prototypes cached to '{PROTOTYPE_CACHE}' -- faster next time.")
print(f"Ready — {len(positive_prototypes)} prototype channels loaded.")


# ─── Inference helper ────────────────────────────────────────────────────────

def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert a CHW float tensor (ImageNet-normalised) back to a PIL image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = t.cpu().float() * std + mean
    t = t.clamp(0, 1)
    return T.ToPILImage()(t)


def _get_bbox_from_feature_map(feature_model, image_tensor, channel, U, device, img_size):
    """Compute yellow box coords by finding argmax of channel k in feature space."""
    with torch.no_grad():
        x = image_tensor.unsqueeze(0).to(device)
        Z = feature_model(x)          # (1, D, H, W)
        # apply disentanglement
        B, D, H, W = Z.shape
        Z_flat = Z.view(B, D, -1).permute(0, 2, 1)   # (1, H*W, D)
        Z_dis  = Z_flat @ U.T                          # (1, H*W, D)
        Z_dis  = Z_dis.permute(0, 2, 1).view(B, D, H, W)

        # argmax over spatial dims for this channel
        ch_map = Z_dis[0, channel]                     # (H, W)
        flat_idx = ch_map.argmax().item()
        fh, fw = flat_idx // W, flat_idx % W

        # scale to image pixels
        ih, iw = img_size
        cell_h, cell_w = ih / H, iw / W
        x1 = int(fw * cell_w)
        y1 = int(fh * cell_h)
        x2 = int(x1 + cell_w)
        y2 = int(y1 + cell_h)
        return x1, y1, x2, y2


def explain(pil_image: Image.Image, top_k: int) -> np.ndarray:
    """
    Run EPIC explanation for a single PIL image.
    Returns a matplotlib figure rendered as an RGB numpy array.
    """
    set_seeds(42)

    # Apply the same transform the model was trained with
    image_tensor = epic_transform(pil_image.convert("RGB"))

    # 1. Find top-k channels most responsible for the prediction
    channels = topk_active_channels(
        feature_model,
        classification_head,
        image_tensor,
        k=top_k,
        device=device,
    )

    # 2. Localise the activated patch within the query image
    image_prototypes = get_image_prototypes(
        feature_model,
        image_tensor,
        channels,
        U=U,
        device=device,
        img_fn="none",
    )

    # 3. Find the nearest training images for each channel
    visualized_prototypes = get_visualized_prototypes(
        feature_model,
        positive_prototypes,
        dataloader_train,
        channels,
        U=U,
        device=device,
        img_fn="none",
    )

    # ─── Build the explanation figure (mirrors paper layout) ─────────────────
    cols = 1 + NUM_PROTOTYPES          # query image + prototype neighbours
    rows = top_k
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    fig.patch.set_facecolor("#1a1a1a")

    if rows == 1:
        axes = [axes]

    for row_idx, ch in enumerate(channels):
        ax_row = axes[row_idx]

        # ── Column 0: query image with highlighted patch ──────────────────
        ax = ax_row[0]
        query_pil = _tensor_to_pil(image_tensor)
        ax.imshow(query_pil)
        ax.set_title(f"Channel {ch}", color="white", fontsize=8, pad=3)
        ax.axis("off")

        # Compute bbox from feature map argmax
        x1, y1, x2, y2 = _get_bbox_from_feature_map(
            feature_model, image_tensor, ch, U, device,
            img_size=(query_pil.size[1], query_pil.size[0])   # (H, W)
        )
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="yellow", facecolor="none"
        )
        ax.add_patch(rect)

        # ── Columns 1…N: training-set prototype images ───────────────────
        proto_imgs = visualized_prototypes.get(ch, [])
        for col_idx in range(1, cols):
            ax = ax_row[col_idx]
            ax.axis("off")
            ax.set_facecolor("#1a1a1a")
            if col_idx - 1 < len(proto_imgs):
                proto_data = proto_imgs[col_idx - 1]
                # proto_data may be a tensor or PIL Image
                if isinstance(proto_data, torch.Tensor):
                    proto_pil = _tensor_to_pil(proto_data)
                else:
                    proto_pil = proto_data
                ax.imshow(proto_pil)

    plt.tight_layout(pad=0.4)

    # Render figure to numpy array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(h, w, 4)[:, :, 1:]
    plt.close(fig)
    return buf


# ─── Gradio UI ────────────────────────────────────────────────────────────────

def gradio_predict(image, top_k):
    if image is None:
        return None
    try:
        result = explain(Image.fromarray(image), int(top_k))
        return result
    except Exception as e:
        # Return a simple error image
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center",
                fontsize=10, color="red", wrap=True)
        ax.axis("off")
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)[:, :, 1:]
        plt.close(fig)
        return buf


with gr.Blocks(
    title="EPIC — Prototype Explanation Demo",
    theme=gr.themes.Soft(),
    css="""
        #title { text-align: center; margin-bottom: 0.5rem; }
        #subtitle { text-align: center; color: #666; margin-bottom: 1.5rem; }
        .explain-btn { background: #f5a623 !important; color: white !important; }
    """,
) as demo:

    gr.Markdown("# 🔍 EPIC — Prototype Explanation", elem_id="title")
    gr.Markdown(
        "Upload an image. EPIC will show which **prototype parts** from the "
        "training set the model relied on to make its prediction.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Input image",
                type="numpy",
                height=280,
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=8, step=1, value=TOP_K,
                label="Number of prototype rows (top-k channels)",
            )
            run_btn = gr.Button("Explain ▶", elem_classes="explain-btn", size="lg")

        with gr.Column(scale=3):
            output_image = gr.Image(
                label="Explanation  —  column 0: query image with activated region  |  columns 1–5: nearest training prototypes",
                type="numpy",
                height=600,
            )

    gr.Markdown(
        "**How to read this:** Each row corresponds to one feature channel "
        "(prototype concept). The yellow box on the left marks where that concept "
        "fires in your image. The images to the right are the training examples "
        "that activate the same channel most strongly.",
    )

    run_btn.click(
        fn=gradio_predict,
        inputs=[input_image, top_k_slider],
        outputs=output_image,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )