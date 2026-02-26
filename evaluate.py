import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from model import TS_CLIP
from dataset import UCRDataset


def evaluate_dataset(
    model: TS_CLIP,
    dataset_name: str,
    ucr_data_path: str = './UCRArchive_2018',
    annotation_path: str = './preprocess_label.json',
    limit_length: int = 200,
    device: torch.device = None,
) -> float:
    """
    Run zero-shot classification on a single UCR dataset and return accuracy.
    The model must already be loaded and moved to device before calling this.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = UCRDataset(
        dataset_name=dataset_name,
        ucr_data_path=ucr_data_path,
        annotation_path=annotation_path,
        limit_length=limit_length,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=UCRDataset.collate_fn)

    correct = 0
    total = 0

    with torch.no_grad():
        for ts, text_embedding, label, annos in dataloader:
            ts = ts.to(device).float()

            # Per-sample z-score normalization
            mean = ts.mean(dim=-1, keepdim=True)
            std = ts.std(dim=-1, keepdim=True)
            ts = (ts - mean) / std

            # Build class prototype embeddings from annotations for this dataset
            class_annos = annos[0]
            ori_class_ids = [int(a['ori_sub_class_id']) for a in class_annos]
            prompt_embeddings = torch.tensor(
                [a['prompt_embedding'] for a in class_annos], dtype=torch.float32
            ).to(device)

            ts_feature, _ = model(ts, None)

            # Cosine similarity between time series feature and each class prototype
            similarities = F.cosine_similarity(
                ts_feature.unsqueeze(1), prompt_embeddings.unsqueeze(0), dim=-1
            )
            pred_idx = similarities.argmax(dim=-1).item()
            pred_class = ori_class_ids[pred_idx]
            true_class = label[0].item()

            if pred_class == true_class:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def load_model(
    checkpoint_path: str,
    model_path: str = './TimeMoE_50M',
    device: torch.device = None,
) -> TS_CLIP:
    """Load TS-CLIP model from checkpoint.

    The checkpoint contains all weights (TimeMoE encoder + projector), so no separate
    TimeMoE pre-trained weights file is required.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TS_CLIP(model_path=model_path)
    ckp = torch.load(checkpoint_path, map_location=device)
    # Strip 'module.' prefix if model was saved with DataParallel
    ckp = {k.replace('module.', ''): v for k, v in ckp.items()}
    model.load_state_dict(ckp)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='TS-CLIP Zero-Shot Evaluation on UCR Archive')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint.pt',
                        help='Path to the model checkpoint (.pt file)')
    parser.add_argument('--ucr_data_path', type=str, default='./UCRArchive_2018',
                        help='Path to the UCR Archive 2018 dataset directory')
    parser.add_argument('--annotation_path', type=str, default='./preprocess_label.json',
                        help='Path to the annotation JSON file')
    parser.add_argument('--model_path', type=str, default='./TimeMoE_50M',
                        help='Path to the TimeMoE_50M model directory (config / architecture only)')
    parser.add_argument('--limit_length', type=int, default=200,
                        help='Maximum time series length (longer series will be downsampled)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='List of dataset names to evaluate. If not specified, all datasets are evaluated.')
    parser.add_argument('--output', type=str, default='./eval_results.csv',
                        help='Path to save evaluation results CSV')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f'Loading model from checkpoint: {args.checkpoint}')
    model = load_model(
        checkpoint_path=args.checkpoint,
        model_path=args.model_path,
        device=device,
    )

    dataset_names = args.datasets if args.datasets else sorted(os.listdir(args.ucr_data_path))

    results = []
    for dataset_name in dataset_names:
        print(f'Evaluating {dataset_name} ...', end=' ', flush=True)
        try:
            acc = evaluate_dataset(
                model=model,
                dataset_name=dataset_name,
                ucr_data_path=args.ucr_data_path,
                annotation_path=args.annotation_path,
                limit_length=args.limit_length,
                device=device,
            )
            print(f'Accuracy: {acc:.4f}')
            results.append((dataset_name, acc))
        except Exception as e:
            print(f'FAILED ({e})')
            results.append((dataset_name, None))

    # Save results to CSV
    with open(args.output, 'w') as f:
        f.write('dataset,accuracy\n')
        for name, acc in results:
            acc_str = f'{acc:.4f}' if acc is not None else 'N/A'
            f.write(f'{name},{acc_str}\n')

    # Print summary table
    valid_results = [(n, a) for n, a in results if a is not None]
    if valid_results:
        mean_acc = np.mean([a for _, a in valid_results])
        print(f'\n{"Dataset":<45} {"Accuracy":>10}')
        print('-' * 57)
        for name, acc in results:
            acc_str = f'{acc:.4f}' if acc is not None else 'N/A'
            print(f'{name:<45} {acc_str:>10}')
        print('-' * 57)
        print(f'{"Mean Accuracy":<45} {mean_acc:>10.4f}')

    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()

