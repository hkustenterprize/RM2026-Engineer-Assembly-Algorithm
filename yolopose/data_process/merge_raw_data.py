"""Merge multiple Blender-rendered datasets into one.

Each input dataset must have:
    images/          — PNG images (00000.png, 00001.png, ...)
    annotations.json — {keypoint_names, keypoint_3d_world, images: [...]}

The script copies (or symlinks) all images into a single output directory with
globally sequential filenames and writes one merged annotations.json.

Usage:

python /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/yolopose/data_process/merge_raw_data.py \
  -o /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/all \
  --datasets \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/1 \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/2 \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/3 \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/4 \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/5 \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/7 \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/8 \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/9 \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/10 \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/11 \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/12 \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/13 \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/14 \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/15 \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/16 \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0/18 

python merge_raw_data.py --datasets dir_a dir_b dir_c -o /path/to/output

    # 使用软链接 (节省磁盘空间, 速度更快)
    python merge_raw_data.py --datasets dir_a dir_b dir_c -o /path/to/output --symlink
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def load_dataset(dataset_dir: str) -> dict:
    ann_path = os.path.join(dataset_dir, "annotations.json")
    with open(ann_path) as f:
        return json.load(f)


def _transfer_file(src: str, dst: str, use_symlink: bool) -> None:
    """Copy or symlink src → dst. 若 dst 已存在则跳过."""
    if os.path.exists(dst):
        return
    if use_symlink:
        os.symlink(os.path.abspath(src), dst)
    else:
        import shutil

        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Merge Blender datasets")
    parser.add_argument(
        "--datasets", nargs="+", help="paths to dataset directories to merge"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="output directory for merged dataset"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="使用软链接代替复制 (节省空间, 仅限同机器)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help="文件传输线程数 (默认 min(16, cpu_count))",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    merged_images: list = []
    keypoint_names: list | None = None
    keypoint_3d_world = None
    targets_meta = None
    global_id = 0

    # ── 第一遍: 收集所有 entry + 建立传输任务 ──
    tasks: list[tuple[str, str]] = []  # (src, dst)

    for ds_path_raw in args.datasets:
        ds_path = ds_path_raw.rstrip("/")
        ds_name = os.path.basename(ds_path)
        ann = load_dataset(ds_path)

        if keypoint_names is None:
            keypoint_names = ann.get("keypoint_names", [])
            keypoint_3d_world = ann.get("keypoint_3d_world")
            if keypoint_3d_world is None:
                for tgt in ann.get("targets", []):
                    if tgt.get("keypoints_3d"):
                        keypoint_3d_world = tgt["keypoints_3d"]
                        break
            targets_meta = ann.get("targets")

        n = len(ann["images"])
        print(f"  {ds_name}: {n} images")

        for entry in ann["images"]:
            old_file = entry["file_name"]
            new_file = f"{global_id:05d}.png"

            src = os.path.join(ds_path, "images", old_file)
            dst = str(out_images / new_file)
            tasks.append((src, dst))

            new_entry = dict(entry)
            new_entry["id"] = global_id
            new_entry["file_name"] = new_file
            if "meta" in new_entry and isinstance(new_entry["meta"], dict):
                new_entry["meta"]["source_dataset"] = ds_name
            merged_images.append(new_entry)
            global_id += 1

    # ── 第二遍: 多线程传输文件 ──
    mode = "symlink" if args.symlink else "copy"
    print(f"\n[merge] {global_id} 张图, {mode} 模式, {args.workers} 线程 ...")
    with tqdm(total=len(tasks), unit="img", dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futs = {
                executor.submit(_transfer_file, src, dst, args.symlink): dst
                for src, dst in tasks
            }
            for fut in as_completed(futs):
                fut.result()
                pbar.update(1)

    # ── 写 annotations.json ──
    merged_ann: dict = {
        "keypoint_names": keypoint_names,
        "keypoint_3d_world": keypoint_3d_world,
        "images": merged_images,
    }
    if targets_meta is not None:
        merged_ann["targets"] = targets_meta

    out_ann = out_dir / "annotations.json"
    with open(out_ann, "w") as f:
        json.dump(merged_ann, f)

    print(
        f"\nMerged {global_id} images from {len(args.datasets)} datasets → {out_dir}/"
    )
    if args.symlink:
        print("  (软链接模式: 删除输出目录不影响原始数据)")


if __name__ == "__main__":
    main()
