import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Paths, TrainConfig
from data import (
    RVLCDIPOCRTextDataset,
    RVLCDIPOODTextDataset,
    set_seed,
    resolve_split_limit,
)
from model import BertDocClassifier
from knn_ood import (
    extract_embeddings_and_logits,
    build_faiss_l2_index,
    estimate_threshold_theta,
    knn_star_predict,
    knn_predict_no_agreement,
    knn1_score_and_neighbor,
)
from metrics import compute_auc, compute_fpr_at_tpr95, compute_end_to_end_metrics


LOSS_ORDER = ["margin", "margin_star", "scl", "weight", "ce"]

LOSS_DISPLAY = {
    "margin": "Margin",
    "margin_star": "Margin*",
    "scl": "SCL",
    "weight": "Weight",
    "ce": "CE",
}


def _parse_samples(val: str):
    if val is None or val.lower() == "full":
        return None
    return int(val)


def find_ckpts(ckpt_dir: Path) -> List[Tuple[str, Path]]:
    found = []
    for loss in LOSS_ORDER:
        p = ckpt_dir / f"bert_{loss}.pt"
        if p.exists():
            found.append((loss, p))
    return found


def compute_scores_preds_nns(train_index, embeddings: np.ndarray, logits: np.ndarray):
    scores = np.zeros((embeddings.shape[0],), dtype=np.float32)
    preds = logits.argmax(axis=1).astype(np.int64)
    nn_labels = np.zeros((embeddings.shape[0],), dtype=np.int64)

    for i in range(embeddings.shape[0]):
        s, _, nn_label = knn1_score_and_neighbor(train_index, embeddings[i])
        scores[i] = s
        nn_labels[i] = nn_label

    return scores, preds, nn_labels


def make_knn_star_scores(scores: np.ndarray, preds: np.ndarray, nn_labels: np.ndarray) -> np.ndarray:
    # If classifier prediction disagrees with the 1NN label, force score below any valid threshold.
    floor = float(scores.min()) - 1e-6
    out = scores.copy()
    out[preds != nn_labels] = floor
    return out


def eval_one_checkpoint(
    project_root: Path,
    ckpt_path: Path,
    cfg: TrainConfig,
    tpr_target: float,
    save_dir: Optional[Path] = None,
) -> dict:
    paths = Paths(
        project_root=project_root,
        qs_ocr_large_dir=project_root / "rvl-cdip-text",
        rvl_cdip_dir=project_root / "rvl-cdip",
        rvl_cdip_ood_text_dir=project_root / "rvl-cdip-o-text",
        train_list=project_root / "train.txt",
        val_list=project_root / "val.txt",
        test_list=project_root / "test.txt",
    )

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model_name = ckpt["model_name"]
    max_length = ckpt["max_length"]
    num_classes = int(ckpt.get("num_classes", 16))

    model = BertDocClassifier(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    train_limit = resolve_split_limit(cfg, "train")
    val_limit = resolve_split_limit(cfg, "val")
    test_limit = resolve_split_limit(cfg, "test")

    train_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.train_list,
        tokenizer_name=model_name,
        max_length=max_length,
        debug_samples=train_limit,
    )
    val_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.val_list,
        tokenizer_name=model_name,
        max_length=max_length,
        debug_samples=val_limit,
    )
    test_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.test_list,
        tokenizer_name=model_name,
        max_length=max_length,
        debug_samples=test_limit,
    )
    ood_ds = RVLCDIPOODTextDataset(
        ood_text_dir=paths.rvl_cdip_ood_text_dir,
        tokenizer_name=model_name,
        max_length=max_length,
        debug_samples=test_limit,
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=cfg.num_workers)
    ood_loader = DataLoader(ood_ds, batch_size=16, shuffle=False, num_workers=cfg.num_workers)

    train_emb, _, train_y = extract_embeddings_and_logits(model, train_loader, device=str(device))
    val_emb, _, _ = extract_embeddings_and_logits(model, val_loader, device=str(device))
    test_emb, test_logits, test_y = extract_embeddings_and_logits(model, test_loader, device=str(device))
    ood_emb, ood_logits, _ = extract_embeddings_and_logits(model, ood_loader, device=str(device))

    train_index = build_faiss_l2_index(train_emb.astype(np.float32), train_y.astype(np.int64))

    theta = estimate_threshold_theta(
        train_index=train_index,
        val_embeddings=val_emb.astype(np.float32),
        tpr_target=tpr_target,
    )

    # KNN scores
    id_scores, id_pred_for_score, id_nn_for_score = compute_scores_preds_nns(
        train_index, test_emb.astype(np.float32), test_logits
    )
    ood_scores, ood_pred_for_score, ood_nn_for_score = compute_scores_preds_nns(
        train_index, ood_emb.astype(np.float32), ood_logits
    )

    # KNN* scores = distance score + agreement gate
    id_scores_star = make_knn_star_scores(id_scores, id_pred_for_score, id_nn_for_score)
    ood_scores_star = make_knn_star_scores(ood_scores, ood_pred_for_score, ood_nn_for_score)

    # Novelty detection metrics
    knn_auc = compute_auc(id_scores, ood_scores) * 100.0
    knn_fpr = compute_fpr_at_tpr95(id_scores, ood_scores, tpr_target=tpr_target) * 100.0

    knn_star_auc = compute_auc(id_scores_star, ood_scores_star) * 100.0
    knn_star_fpr = compute_fpr_at_tpr95(id_scores_star, ood_scores_star, tpr_target=tpr_target) * 100.0

    # End-to-end pipeline metrics
    is_id_test_knn, pred_knn = knn_predict_no_agreement(train_index, test_emb, test_logits, theta)
    is_id_ood_knn, _ = knn_predict_no_agreement(train_index, ood_emb, ood_logits, theta)

    is_id_test_knn_star, pred_knn_star = knn_star_predict(train_index, test_emb, test_logits, theta)
    is_id_ood_knn_star, _ = knn_star_predict(train_index, ood_emb, ood_logits, theta)

    knn_m = compute_end_to_end_metrics(test_y, is_id_test_knn, pred_knn, is_id_ood_knn)
    knn_star_m = compute_end_to_end_metrics(test_y, is_id_test_knn_star, pred_knn_star, is_id_ood_knn_star)

    stem = ckpt_path.stem
    artifacts = {
        "id_scores_knn": f"{stem}_id_scores_knn.npy",
        "ood_scores_knn": f"{stem}_ood_scores_knn.npy",
        "id_scores_knn_star": f"{stem}_id_scores_knn_star.npy",
        "ood_scores_knn_star": f"{stem}_ood_scores_knn_star.npy",
        "id_pred_labels": f"{stem}_id_pred_labels.npy",
        "id_nn_labels": f"{stem}_id_nn_labels.npy",
        "ood_pred_labels": f"{stem}_ood_pred_labels.npy",
        "ood_nn_labels": f"{stem}_ood_nn_labels.npy",
        "id_test_y_true": f"{stem}_id_test_y_true.npy",
        "theta": theta,
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / artifacts["id_scores_knn"], id_scores.astype(np.float32))
        np.save(save_dir / artifacts["ood_scores_knn"], ood_scores.astype(np.float32))
        np.save(save_dir / artifacts["id_scores_knn_star"], id_scores_star.astype(np.float32))
        np.save(save_dir / artifacts["ood_scores_knn_star"], ood_scores_star.astype(np.float32))

        np.save(save_dir / artifacts["id_pred_labels"], id_pred_for_score.astype(np.int64))
        np.save(save_dir / artifacts["id_nn_labels"], id_nn_for_score.astype(np.int64))
        np.save(save_dir / artifacts["ood_pred_labels"], ood_pred_for_score.astype(np.int64))
        np.save(save_dir / artifacts["ood_nn_labels"], ood_nn_for_score.astype(np.int64))
        np.save(save_dir / artifacts["id_test_y_true"], test_y.astype(np.int64))

        metrics_payload = {
            "checkpoint": str(ckpt_path),
            "split": {
                "train_samples": train_limit if train_limit is not None else "full",
                "val_samples": val_limit if val_limit is not None else "full",
                "test_samples": test_limit if test_limit is not None else "full",
            },
            "tpr_target": tpr_target,
            "theta": float(theta),
            "knn": {
                "fpr_at_tpr": knn_fpr,
                "auc": knn_auc,
                "f1": knn_m.f1 * 100.0,
                "cov": knn_m.cov * 100.0,
            },
            "knn_star": {
                "fpr_at_tpr": knn_star_fpr,
                "auc": knn_star_auc,
                "f1": knn_star_m.f1 * 100.0,
                "cov": knn_star_m.cov * 100.0,
            },
            "artifacts": artifacts,
        }

        with open(save_dir / f"{stem}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

    return {
        "knn_fpr": knn_fpr,
        "knn_auc": knn_auc,
        "knn_f1": knn_m.f1 * 100.0,
        "knn_cov": knn_m.cov * 100.0,
        "knn_star_fpr": knn_star_fpr,
        "knn_star_auc": knn_star_auc,
        "knn_star_f1": knn_star_m.f1 * 100.0,
        "knn_star_cov": knn_star_m.cov * 100.0,
        "artifacts": artifacts,
    }


def _print_table(title: str, results: list, auc_key: str, fpr_key: str, f1_key: str, cov_key: str):
    header = f"{'#':<3} {'loss':<10} {'FPR↓':>8} {'AUC↑':>8} {'F1↑':>8} {'COV↑':>8}"
    sep = "-" * len(header)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for i, (loss, m) in enumerate(results, start=1):
        print(
            f"{i:<3} {LOSS_DISPLAY.get(loss, loss):<10} "
            f"{m[fpr_key]:>8.2f} {m[auc_key]:>8.2f} {m[f1_key]:>8.2f} {m[cov_key]:>8.2f}"
        )
    print(sep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--ckpt", type=str, default=None, help="Path to a single checkpoint .pt file.")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Scan dir for all bert_*.pt checkpoints.")
    ap.add_argument(
        "--use",
        type=str,
        default=None,
        choices=["cpu", "gpu"],
        help="Device: 'cpu' or 'gpu'.",
    )
    ap.add_argument(
        "--tpr",
        type=float,
        default=0.95,
        help="TPR target for threshold estimation, e.g. 0.95 (default), 0.90, 0.99.",
    )
    ap.add_argument("--train_samples", type=str, default=None, help="Training samples or 'full'.")
    ap.add_argument("--val_samples", type=str, default=None, help="Validation samples or 'full'.")
    ap.add_argument("--test_samples", type=str, default=None, help="Test/OOD samples or 'full'.")
    ap.add_argument("--save_dir", type=str, default="novelty_outputs_text")
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.use is not None:
        cfg.device = "cuda" if args.use == "gpu" else "cpu"
    if args.train_samples is not None or args.val_samples is not None or args.test_samples is not None:
        cfg.debug_samples = None
    if args.train_samples is not None:
        cfg.train_samples = _parse_samples(args.train_samples)
    if args.val_samples is not None:
        cfg.val_samples = _parse_samples(args.val_samples)
    if args.test_samples is not None:
        cfg.test_samples = _parse_samples(args.test_samples)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
        loss = ckpt_path.stem.removeprefix("bert_")
        ckpts = [(loss, ckpt_path)]
    else:
        ckpt_dir = Path(args.ckpt_dir)
        ckpts = find_ckpts(ckpt_dir)
        if not ckpts:
            print(f"No checkpoints found in {ckpt_dir}.")
            print("Expected names: bert_margin_star.pt, bert_ce.pt, etc.")
            return

    results = []
    for loss, ckpt_path in ckpts:
        print(f"\n--- Evaluating {loss} ({ckpt_path.name}) ---")
        m = eval_one_checkpoint(
            Path(args.project_root),
            ckpt_path,
            cfg,
            tpr_target=args.tpr,
            save_dir=save_dir,
        )
        results.append((loss, m))

    results.sort(key=lambda x: LOSS_ORDER.index(x[0]))

    tpr_pct = int(args.tpr * 100)
    _print_table(f"KNN     (FPR@TPR{tpr_pct})", results, "knn_auc", "knn_fpr", "knn_f1", "knn_cov")
    _print_table(f"KNN*    (FPR@TPR{tpr_pct})", results, "knn_star_auc", "knn_star_fpr", "knn_star_f1", "knn_star_cov")

    print("\nNotes:")
    print(f"  FPR = FPR@TPR{tpr_pct}  |  AUC from ID-test vs OOD scores")
    print("  KNN uses raw 1NN distance score")
    print("  KNN* uses raw 1NN distance score gated by consensus agreement (pred == 1NN label)")
    print(f"  Saved raw OOD score artifacts to: {save_dir}")


if __name__ == "__main__":
    main()