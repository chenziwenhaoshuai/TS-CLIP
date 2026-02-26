"""
demo.py — Interactive zero-shot classification demo for TS-CLIP.

Pick any two UCR datasets and supply one text label per dataset.
The script encodes both labels with the CLIP text encoder, then uses the
TS-CLIP time series encoder to classify which series matches which label.

Example
-------
python demo.py \
    --datasets   BeetleFly Car \
    --labels     "beetle fly" "car engine" \
    --sample_idx 0 0
"""

import argparse
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import clip
from dataset import UCRDataset
from model import TS_CLIP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, model_path: str, device: torch.device) -> TS_CLIP:
    model = TS_CLIP(model_path=model_path)
    ckp = torch.load(checkpoint_path, map_location=device)
    ckp = {k.replace('module.', ''): v for k, v in ckp.items()}
    model.load_state_dict(ckp)
    model.to(device)
    model.eval()
    return model


def normalize(ts: torch.Tensor) -> torch.Tensor:
    """Per-sample z-score normalization."""
    mean = ts.mean(dim=-1, keepdim=True)
    std  = ts.std(dim=-1, keepdim=True)
    return (ts - mean) / std


def get_sample(dataset_name: str, sample_idx: int,
               ucr_data_path: str, annotation_path: str,
               limit_length: int, device: torch.device) -> Tuple[torch.Tensor, int]:
    """Load a single time series from a UCR dataset, return shape (1, T)."""
    ds = UCRDataset(
        dataset_name=dataset_name,
        ucr_data_path=ucr_data_path,
        annotation_path=annotation_path,
        limit_length=limit_length,
    )
    ts, _, label, _ = ds[sample_idx]
    print(f"  [{dataset_name}] sample #{sample_idx}  true label = {label.item()}")
    return ts.unsqueeze(0).to(device).float(), label.item()


def plot_series(ts_list, dataset_names, pred_labels, true_labels, save_path=None):
    """Plot the two time series side by side with prediction results."""
    n = len(ts_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 2.5))
    if n == 1:
        axes = [axes]
    for ax, ts, name, pred, true in zip(axes, ts_list, dataset_names, pred_labels, true_labels):
        ax.plot(ts)
        ax.set_title(f"{name}\nPredicted: {pred}  (true label id: {true})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='TS-CLIP demo: classify two time series with free-text labels.'
    )
    parser.add_argument('--datasets', type=str, nargs=2, required=True,
                        metavar=('DATASET_A', 'DATASET_B'),
                        help='Two UCR dataset names, e.g. BeetleFly Car')
    parser.add_argument('--labels', type=str, nargs=2, required=True,
                        metavar=('LABEL_A', 'LABEL_B'),
                        help='Two text labels to classify against, e.g. "beetle fly" "car engine"')
    parser.add_argument('--sample_idx', type=int, nargs=2, default=[0, 0],
                        metavar=('IDX_A', 'IDX_B'),
                        help='Which test sample to pick from each dataset (default: 0 0)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint.pt')
    parser.add_argument('--model_path', type=str, default='./TimeMoE_50M')
    parser.add_argument('--clip_weights', type=str, default='./ViT-B-32.pt',
                        help='Path to CLIP ViT-B/32 weights')
    parser.add_argument('--ucr_data_path', type=str, default='./UCRArchive_2018')
    parser.add_argument('--annotation_path', type=str, default='./preprocess_label.json')
    parser.add_argument('--limit_length', type=int, default=200)
    parser.add_argument('--save_fig', type=str, default=None,
                        help='Optional path to save the output figure, e.g. result.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # ------------------------------------------------------------------
    # 1. Load TS-CLIP
    # ------------------------------------------------------------------
    print('Loading TS-CLIP model ...')
    ts_clip = load_model(args.checkpoint, args.model_path, device)

    # ------------------------------------------------------------------
    # 2. Load CLIP text encoder
    # ------------------------------------------------------------------
    print('Loading CLIP text encoder ...')
    clip_model, _ = clip.load(args.clip_weights, device=device)
    clip_model.eval()

    # ------------------------------------------------------------------
    # 3. Encode text labels → (2, 512) text features
    # ------------------------------------------------------------------
    print(f'Encoding text labels: {args.labels}')
    tokens = clip.tokenize(args.labels).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens).float()   # (2, 512)

    # ------------------------------------------------------------------
    # 4. Load one sample from each dataset
    # ------------------------------------------------------------------
    print('\nLoading time series samples:')
    ts_tensors, true_labels, raw_arrays = [], [], []
    for name, idx in zip(args.datasets, args.sample_idx):
        ts, true_label = get_sample(
            name, idx, args.ucr_data_path, args.annotation_path, args.limit_length, device
        )
        ts_tensors.append(ts)
        true_labels.append(true_label)
        raw_arrays.append(normalize(ts).squeeze().cpu().numpy())

    # ------------------------------------------------------------------
    # 5. Classify each series against the two text labels
    # ------------------------------------------------------------------
    print('\n--- Classification Results ---')
    predicted_labels = []
    with torch.no_grad():
        for ts, name in zip(ts_tensors, args.datasets):
            ts_norm = normalize(ts)
            ts_feat, _ = ts_clip(ts_norm, None)           # (1, 512)

            sims = F.cosine_similarity(
                ts_feat.unsqueeze(1),                     # (1, 1, 512)
                text_features.unsqueeze(0),               # (1, 2, 512)
                dim=-1                                    # → (1, 2)
            )
            probs = F.softmax(sims, dim=-1).squeeze()     # (2,)
            pred_idx = probs.argmax().item()
            pred_label = args.labels[pred_idx]
            predicted_labels.append(pred_label)

            print(f"  {name}:")
            for i, (lbl, p) in enumerate(zip(args.labels, probs.tolist())):
                marker = ' ◀' if i == pred_idx else ''
                print(f"    '{lbl}': {p:.4f}{marker}")

    # ------------------------------------------------------------------
    # 6. Visualise
    # ------------------------------------------------------------------
    plot_series(raw_arrays, args.datasets, predicted_labels, true_labels, save_path=args.save_fig)


if __name__ == '__main__':
    main()

