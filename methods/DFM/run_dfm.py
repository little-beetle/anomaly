import os
import torch
import json
import csv
from anomalib.data import MVTec
from anomalib.models import Dfm
from anomalib.engine import Engine

if __name__ == "__main__":
    # -----------------------
    # ⚙️ DEVICE
    # -----------------------
    device = "cpu"
    print(f"⚙️ Using device: {device}")

    # -----------------------
    # 📂 PATHS
    # -----------------------
    project_root = os.getcwd()
    dataset_root = os.path.join(project_root, "datasets", "mvtec")
    results_dir = os.path.join(project_root, "methods", "DFM", "results")
    os.makedirs(results_dir, exist_ok=True)

    # -----------------------
    # 🧩 DATA MODULE
    # -----------------------
    category = "bottle"  # 👉 змінюй за потреби
    datamodule = MVTec(
        root=dataset_root,
        category=category,
        image_size=(256, 256),
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0,
    )
    datamodule.prepare_data()
    datamodule.setup()

    # -----------------------
    # 🧠 MODEL + ENGINE
    # -----------------------
    model = Dfm(backbone="resnet18")

    engine = Engine(
        max_epochs=1,
        accelerator=device,
        devices=1,
        logger=False,
        default_root_dir=results_dir,
    )

    # -----------------------
    # 🚀 TRAIN + TEST
    # -----------------------
    print("🚀 Starting training...")
    engine.fit(model=model, datamodule=datamodule)

    print("🧪 Running test...")
    metrics_list = engine.test(model=model, datamodule=datamodule)
    metrics = metrics_list[0] if metrics_list else {}
    print("✅ Done! Test metrics:")
    print(metrics)

    # -----------------------
    # 💾 SAVE RESULTS
    # -----------------------
    json_path = os.path.join(results_dir, f"{category}_metrics.json")
    csv_path = os.path.join(results_dir, f"{category}_metrics.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, round(float(v), 4)])

    print(f"🎯 Results saved in:\n  • {json_path}\n  • {csv_path}")
