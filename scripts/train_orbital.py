"""Train the learned Orbital landmark classifier on real labelled HiRISE images.

Data: "Mars orbital image (HiRISE) labeled data set version 3.2" (Wagstaff et al.,
NASA JPL; Zenodo 10.5281/zenodo.4002935, CC BY 4.0). 227x227 grayscale landmark crops
from 232 HiRISE browse images. The published train/val/test split separates source
HiRISE images; augmented copies stay in their original's split. Model selection uses
validation macro-F1 only; the test split is evaluated once at the end.
Classes: other, crater, dark dune, slope streak, bright dune, impact ejecta, swiss cheese, spider.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

_dll_handle = os.add_dll_directory(os.environ["MARS_DLL_DIR"]) if os.name == "nt" and os.environ.get("MARS_DLL_DIR") else None
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "prepared"
CLASSES = ["other", "crater", "dark_dune", "slope_streak", "bright_dune", "impact_ejecta", "swiss_cheese", "spider"]
SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class Landmarks(Dataset):
    def __init__(self, items, root, train):
        self.items, self.root, self.train = items, root, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, label = self.items[i]
        im = Image.open(self.root / name).convert("L").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        x = np.asarray(im, dtype=np.float32) / 255
        if self.train and random.random() < .5:
            x = x * random.uniform(.85, 1.15) + random.uniform(-.06, .06)
        x = np.clip(x, 0, 1)
        x = np.stack([(x - m) / s for m, s in zip(MEAN, STD)])
        return torch.from_numpy(x.astype(np.float32)), label


def scores(conf):
    tp = np.diag(conf).astype(float); p = tp / np.maximum(conf.sum(0), 1); r = tp / np.maximum(conf.sum(1), 1)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0)
    return {"accuracy": float(tp.sum() / conf.sum()), "macro_f1": float(f1.mean()),
            "precision": dict(zip(CLASSES, p.round(4).tolist())), "recall": dict(zip(CLASSES, r.round(4).tolist())),
            "f1": dict(zip(CLASSES, f1.round(4).tolist())), "support": dict(zip(CLASSES, conf.sum(1).astype(int).tolist())),
            "confusion_matrix": conf.astype(int).tolist()}


def evaluate(model, loader, device):
    model.eval(); conf = np.zeros((8, 8), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            np.add.at(conf, (y.numpy(), pred), 1)
    return scores(conf)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--export-only", action="store_true", help="evaluate/export the saved best checkpoint without training")
    args = ap.parse_args()
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    lines = [l.split() for l in (args.data / "labels-map-proj_v3_2_train_val_test.txt").read_text().splitlines() if l.strip()]
    root = args.data / "map-proj-v3_2"
    present = {p.name for p in root.iterdir()}
    missing = [n for n, _, _ in lines if n not in present]
    split = {k: [(n, int(c)) for n, c, s in lines if s == k and n in present] for k in ("train", "val", "test")}
    print(f"label rows {len(lines)}, missing image files {len(missing)}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    counts = np.bincount([c for _, c in split["train"]], minlength=8)
    weights = torch.tensor((counts.sum() / (8 * np.maximum(counts, 1))) ** .5, dtype=torch.float32, device=device)
    dl = lambda k, tr: DataLoader(Landmarks(split[k], root, tr), batch_size=48, shuffle=tr, num_workers=4, pin_memory=True, persistent_workers=True)
    train, val, test = dl("train", True), dl("val", False), dl("test", False)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 8)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.epochs * len(train))
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=.05)
    ckpt = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "mars-pathfinder" / "training" / "best-orbital.pt"
    history, best, began = [], -1, time.time()
    if args.export_only:
        history = json.loads((ckpt.parent / "orbital-history.json").read_text())
    for epoch in ([] if args.export_only else range(args.epochs)):
        model.train(); total = 0.0
        for x, y in train:
            opt.zero_grad(); loss = loss_fn(model(x.to(device)), y.to(device)); loss.backward(); opt.step(); sched.step(); total += loss.item()
        v = evaluate(model, val, device)
        history.append({"epoch": epoch + 1, "train_loss": total / len(train), "val_macro_f1": v["macro_f1"], "val_accuracy": v["accuracy"], "elapsed_s": round(time.time() - began, 1)})
        print(json.dumps(history[-1]), flush=True)
        if v["macro_f1"] > best:
            best = v["macro_f1"]; torch.save(model.state_dict(), ckpt)
        (ckpt.parent / "orbital-history.json").write_text(json.dumps(history, indent=1))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    val_scores = evaluate(model, val, device)
    test_scores = evaluate(model, test, device)
    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3, SIZE, SIZE), str(OUT / "orbital.onnx"), opset_version=17, input_names=["image"],
                      output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False, external_data=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT / "orbital.onnx"), providers=["CPUExecutionProvider"])
    x, _ = next(iter(test)); ref = model(x).argmax(1).numpy(); got = sess.run(None, {"image": x.numpy()})[0].argmax(1)
    card = {"ready": True, "name": "MobileNetV3-Small · HiRISE landmark classifier", "task": "8-class orbital landmark classification",
            "classes": CLASSES, "input_size": SIZE, "input": "grayscale replicated to 3 channels, ImageNet normalisation",
            "dataset": "Mars orbital image (HiRISE) labeled data set v3.2", "dataset_url": "https://zenodo.org/records/4002935",
            "dataset_license": "CC-BY-4.0", "citation": "Wagstaff, Lu, Dunkel, McLaughlin, Altinok, Stanboli et al., NASA JPL",
            "split_counts": {k: len(v) for k, v in split.items()}, "label_rows_without_image": len(missing), "split_method": "Published split by source HiRISE image; augmented copies stay with originals",
            "selection": "best validation macro-F1", "epochs": len(history), "seed": 42, "device": device,
            "elapsed_training_s": history[-1]["elapsed_s"] if history else None, "history": history,
            "validation": val_scores, "test": test_scores, "onnx_test_batch_agreement": float((ref == got).mean()),
            "limitations": ["Landmark crops are centred on labelled features at varied browse scales; sliding-window use on a map ortho is a domain shift.",
                            "Class 'other' dominates; rare classes have small test support.",
                            "Classifies 8 landmark types; it does not detect rocks or trafficability."]}
    (OUT / "orbital-model-card.json").write_text(json.dumps(card, indent=2), "utf-8")
    print(json.dumps({"test": {k: test_scores[k] for k in ("accuracy", "macro_f1")}, "val_macro_f1": val_scores["macro_f1"]}))


if __name__ == "__main__":
    main()
