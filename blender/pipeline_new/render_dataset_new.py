"""
render_dataset_new.py — 固定相机 + 物体变换 合成数据生成

驱动模块: pipeline_new/ (固定相机, 物体整体旋转+平移)

用法:
  blender -b exchange_v2.0.blend -P render_dataset_new.py -- \\
        --config ../configs/exchange_new.yaml --n_images 100 --output_dir ./out
        
CUDA_VISIBLE_DEVICES=0 blender -b /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/blender/exchange_removed.blend -P /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/blender/scripts/render_dataset_new.py -- --config /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/blender/configs/exchange_new.yaml --strip_light_color red --output_dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v9.2/front_red --n_images 800 --light_type red

CUDA_VISIBLE_DEVICES=5 blender -b /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/blender/exchange_new.blend -P /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/blender/scripts/render_dataset.py -- --config /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/blender/configs/exchange.yaml --output_dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data/new_blender_test --n_images 10 --light_type red

输出:
  output_dir/
    images/          *.png
    annotations.json COCO 风格标注
    config.yaml      配置副本
    meta.json        运行参数
"""
import bpy
import json
import os
import sys
import random
import datetime
import numpy as np

# ─────────────────────────────────────────────────
#  路径 & 导入
# ─────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPT_PARENT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _SCRIPT_PARENT not in sys.path:
    sys.path.insert(0, _SCRIPT_PARENT)

# ─────────────────────────────────────────────────
#  CLI 参数
# ─────────────────────────────────────────────────
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

import argparse

parser = argparse.ArgumentParser(description="固定相机 + 物体变换 合成数据生成")
parser.add_argument("--config",     type=str, default="../configs/exchange_new.yaml")
parser.add_argument("--n_images",   type=int, default=100)
parser.add_argument("--output_dir", type=str, default="./dataset")
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--samples",    type=int, default=None)
parser.add_argument("--width",      type=int, default=None)
parser.add_argument("--height",     type=int, default=None)
parser.add_argument("--light_type", type=str, default=None,
                    choices=["blue", "red", "off", "random"],
                    help="覆盖 config.lighting.light_type")
parser.add_argument("--strip_light_color", type=str, default=None,
                    choices=["blue", "red", "off", "random"],
                    help="覆盖 config.lighting.strip_lights.color")
parser.add_argument("--strip_light_strength", type=float, default=None,
                    help="覆盖 config.lighting.strip_lights.strength")
parser.add_argument("--strip_light_on_prob", type=float, default=None,
                    help="覆盖 config.lighting.strip_lights.on_prob")
parser.add_argument("--n_shards",   type=int, default=1)
parser.add_argument("--shard_id",   type=int, default=0)
args = parser.parse_args(argv)

# ─────────────────────────────────────────────────
#  加载 YAML 配置
# ─────────────────────────────────────────────────
import yaml

config_path = (
    os.path.join(_SCRIPT_DIR, args.config)
    if not os.path.isabs(args.config)
    else args.config
)
config_path = os.path.abspath(config_path)
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if args.samples is not None:
    cfg["render"]["samples"] = args.samples
if args.width is not None:
    cfg["render"]["width"] = args.width
if args.height is not None:
    cfg["render"]["height"] = args.height

light_type = args.light_type or cfg.get("lighting", {}).get("light_type", "red")
if args.light_type is not None:
    cfg.setdefault("lighting", {})["light_type"] = light_type

strip_cfg = cfg.setdefault("lighting", {}).setdefault("strip_lights", {})
strip_light_color = strip_cfg.get("color")
if args.strip_light_color is not None:
    strip_light_color = args.strip_light_color
    strip_cfg["color"] = strip_light_color

strip_light_strength = strip_cfg.get("strength")
if args.strip_light_strength is not None:
    strip_light_strength = args.strip_light_strength
    strip_cfg["strength"] = strip_light_strength
    strip_cfg.pop("strength_range", None)

strip_light_on_prob = strip_cfg.get("on_prob")
if args.strip_light_on_prob is not None:
    strip_light_on_prob = args.strip_light_on_prob
    strip_cfg["on_prob"] = strip_light_on_prob

render_w = cfg["render"]["width"]
render_h = cfg["render"]["height"]

# ─────────────────────────────────────────────────
#  Shard 分片
# ─────────────────────────────────────────────────
total = args.n_images
shard_size = total // args.n_shards
start_idx  = args.shard_id * shard_size
end_idx    = (start_idx + shard_size
              if args.shard_id < args.n_shards - 1 else total)
n_this_shard = end_idx - start_idx

# ─────────────────────────────────────────────────
#  种子
# ─────────────────────────────────────────────────
shard_seed = args.seed + args.shard_id * 10000
random.seed(shard_seed)
np.random.seed(shard_seed)

# ─────────────────────────────────────────────────
#  输出目录
# ─────────────────────────────────────────────────
os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
with open(os.path.join(args.output_dir, "config.yaml"), "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

run_meta = {
    "timestamp":   datetime.datetime.now().isoformat(),
    "seed":        args.seed,
    "shard_seed":  shard_seed,
    "shard_id":    args.shard_id,
    "n_shards":    args.n_shards,
    "n_images":    args.n_images,
    "light_type":  light_type,
    "strip_light_color": strip_light_color,
    "strip_light_strength": strip_light_strength,
    "strip_light_on_prob": strip_light_on_prob,
    "config_file": os.path.basename(config_path),
    "pipeline":    "pipeline_new (fixed camera + object transform)",
}
if args.shard_id == 0:
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────
#  Blender 场景配置
# ─────────────────────────────────────────────────
scene = bpy.context.scene
scene.render.engine               = cfg["render"].get("engine", "CYCLES")
scene.render.resolution_x         = render_w
scene.render.resolution_y         = render_h
scene.render.resolution_percentage = 100
scene.cycles.samples              = cfg["render"]["samples"]
scene.cycles.use_denoising        = cfg["render"].get("use_denoising", True)
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent     = True
scene.render.image_settings.color_mode  = "RGBA"

_render_dev = cfg["render"].get("device", "GPU").upper()
if _render_dev == "GPU":
    def _enable_gpu(device_type="OPTIX"):
        prefs = bpy.context.preferences
        cp = prefs.addons["cycles"].preferences
        cp.refresh_devices()
        for d in cp.get_devices_for_type(compute_device_type=device_type):
            sys.stdout.write(f"Found device: {d.name} type={getattr(d, 'type', '?')}\n")
            if getattr(d, "type", "").upper() != device_type.upper():
                d.use = False
            else:
                d.use = True
                sys.stdout.write(f"Enabling device: {d.name}\n")
        cp.compute_device_type = device_type
        bpy.context.scene.cycles.device = "GPU"
    try:
        _enable_gpu("OPTIX")
    except Exception as e:
        sys.stdout.write(f"[render] WARN: GPU setup failed ({e})\n")
else:
    scene.cycles.device = "CPU"

# ── 相机 ──
if "RenderCam" not in bpy.data.objects:
    cam_data = bpy.data.cameras.new("RenderCam")
    cam_obj  = bpy.data.objects.new("RenderCam", cam_data)
    scene.collection.objects.link(cam_obj)
else:
    cam_obj  = bpy.data.objects["RenderCam"]
    cam_data = cam_obj.data

scene.camera         = cam_obj
cam_data.type        = "PERSP"
cam_data.sensor_width = cfg["render"].get("sensor_width_mm", 36.0)

# ─────────────────────────────────────────────────
#  Segmap: Object Index Pass
# ─────────────────────────────────────────────────
import fnmatch

def _resolve_mesh_objects(mesh_cfg, all_objects):
    if isinstance(mesh_cfg, dict):
        includes = mesh_cfg.get("include", [])
        excludes = mesh_cfg.get("exclude", [])
    else:
        includes = mesh_cfg if mesh_cfg else []
        excludes = []
    matched = set()
    for pat in includes:
        for obj in all_objects:
            if obj.type == "MESH" and fnmatch.fnmatch(obj.name, pat):
                matched.add(obj)
    for pat in excludes:
        for obj in list(matched):
            if fnmatch.fnmatch(obj.name, pat):
                matched.discard(obj)
    return matched

_targets_cfg = cfg.get("targets", [])
_segmap_dir  = os.path.join(args.output_dir, "images")

_all_blender_objects = list(bpy.data.objects)
for tgt in reversed(_targets_cfg):
    tgt_pass = tgt["pass_index"]
    tgt_objs = _resolve_mesh_objects(tgt.get("mesh_objects", []), _all_blender_objects)
    for obj in tgt_objs:
        obj.pass_index = tgt_pass
for tgt in _targets_cfg:
    tgt_pass = tgt["pass_index"]
    tgt_objs = [o for o in _all_blender_objects if o.type == "MESH" and o.pass_index == tgt_pass]
    sys.stdout.write(f"[render] target '{tgt['name']}': {len(tgt_objs)} mesh → pass_index={tgt_pass}\n")

view_layer = scene.view_layers[0]
view_layer.use_pass_object_index = True

scene.use_nodes = True
_comp_tree = scene.node_tree
_comp_tree.nodes.clear()
_rl_node   = _comp_tree.nodes.new("CompositorNodeRLayers")
_comp_node = _comp_tree.nodes.new("CompositorNodeComposite")
_comp_tree.links.new(_rl_node.outputs["Image"], _comp_node.inputs["Image"])

def _add_target_mask_output(tree, rl_node, pass_index_val, output_dir, slot_prefix):
    gt = tree.nodes.new("CompositorNodeMath")
    gt.operation = "GREATER_THAN"
    gt.inputs[1].default_value = pass_index_val - 0.5
    tree.links.new(rl_node.outputs["IndexOB"], gt.inputs[0])
    lt = tree.nodes.new("CompositorNodeMath")
    lt.operation = "LESS_THAN"
    lt.inputs[1].default_value = pass_index_val + 0.5
    tree.links.new(rl_node.outputs["IndexOB"], lt.inputs[0])
    mul = tree.nodes.new("CompositorNodeMath")
    mul.operation = "MULTIPLY"
    tree.links.new(gt.outputs[0], mul.inputs[0])
    tree.links.new(lt.outputs[0], mul.inputs[1])
    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.base_path = output_dir
    fo.file_slots[0].path = slot_prefix
    fo.format.file_format = "PNG"
    fo.format.color_mode  = "BW"
    fo.format.color_depth = "8"
    tree.links.new(mul.outputs[0], fo.inputs[0])
    return fo

_target_seg_nodes = {}
for tgt in _targets_cfg:
    _prefix = f"seg_{tgt['name']}_"
    _fo = _add_target_mask_output(_comp_tree, _rl_node, tgt["pass_index"], _segmap_dir, _prefix)
    _target_seg_nodes[tgt["name"]] = (_fo, _prefix)
    sys.stdout.write(f"[render] compositor: target '{tgt['name']}' → {_prefix}<frame>.png\n")

# ─────────────────────────────────────────────────
#  构建 pipeline_new Ops
# ─────────────────────────────────────────────────
from pipeline_new import build_ops, RenderContext
from pipeline_new.utils import (
    project_keypoints,
    get_out_of_view_kps,
    get_camera_K,
    blender_cam_to_opencv,
    check_occlusion_raycasts,
)
import cv2 as _cv2_seg

pre_render_ops, post_render_ops, _scene_cfg = build_ops(cfg)

# 收集 target 级别的 min_visible 配置
_min_visible_cfg = cfg.get("min_visible", {})
_occlusion_offset = cfg.get("occlusion_offset_m", 0.001)

sys.stdout.write(f"[render] pipeline_new: {len(pre_render_ops)} pre-render ops, "
                 f"{len(post_render_ops)} post-render ops\n")
sys.stdout.write(f"[render] min_visible: {_min_visible_cfg}\n")

# ─────────────────────────────────────────────────
#  Segmap → bbox 处理
# ─────────────────────────────────────────────────

def _process_segmaps(seg_frame_str, target_seg_nodes, targets_cfg, segmap_dir,
                     W, H):
    """从 segmap 提取每个 target 的 bbox. 返回 dict 或 None (有 target 缺 bbox 则返回 None)."""
    target_bboxes = {}
    for tgt_name, (fo_node, prefix) in target_seg_nodes.items():
        seg_path = os.path.join(segmap_dir, f"{prefix}{seg_frame_str}.png")
        if os.path.exists(seg_path):
            seg_img = _cv2_seg.imread(seg_path, _cv2_seg.IMREAD_GRAYSCALE)
            if seg_img is not None:
                ys, xs = np.where(seg_img > 0)
                if len(xs) > 0:
                    target_bboxes[tgt_name] = [
                        int(xs.min()), int(ys.min()),
                        int(xs.max()), int(ys.max())]
            os.remove(seg_path)

    for tgt in targets_cfg:
        union_names = tgt.get("bbox_from_targets")
        if not union_names:
            continue
        constituent = [target_bboxes[n] for n in union_names if n in target_bboxes]
        if constituent:
            target_bboxes[tgt["name"]] = [
                min(b[0] for b in constituent), min(b[1] for b in constituent),
                max(b[2] for b in constituent), max(b[3] for b in constituent)]

    targets_ann = {}
    for tgt in targets_cfg:
        bb = target_bboxes.get(tgt["name"])
        if bb is None:
            return None
        targets_ann[tgt["name"]] = {"bbox_2d": bb}
    return targets_ann


# ─────────────────────────────────────────────────
#  主循环
# ─────────────────────────────────────────────────
annotations = []
idx = start_idx

while idx < end_idx:
    ctx = RenderContext(scene, cam_obj, cam_data, cfg, light_type=light_type)
    ctx.frame_idx = idx

    # ── Pre-render Ops ──
    for op in pre_render_ops:
        op(ctx)

    # ctx.kp_3d_frame 由 ObjectTransformOp 写入: {tgt_name: [np.array, ...]}
    kp_3d_frame = ctx.kp_3d_frame or {}

    # ── 对每个 target 做遮挡预检 ──
    precheck_ok = True
    all_kps_2d = {}

    for tgt in _targets_cfg:
        tgt_name = tgt["name"]
        kps_3d = kp_3d_frame.get(tgt_name)
        if not kps_3d:
            continue

        kp_names_t = list(tgt.get("keypoints", {}).keys())
        kps_2d = project_keypoints(
            kps_3d, ctx.K, ctx.R_w2c, ctx.t_w2c, ctx.render_w, ctx.render_h)

        oob = get_out_of_view_kps(kp_names_t, kps_2d)
        if oob:
            print(f"[render] frame {idx}: {tgt_name} kps out of view {oob}, retrying...")
            precheck_ok = False
            break

        check_occlusion_raycasts(
            kps_2d, kps_3d, ctx.cam_pos,
            scene=scene, offset_m=_occlusion_offset)

        min_vis = _min_visible_cfg.get(tgt_name, tgt.get("min_visible", 2))
        n_vis = sum(1 for kp in kps_2d if kp[2] == 2)
        if n_vis < min_vis:
            print(f"[render] frame {idx}: {tgt_name} only {n_vis}/{len(kps_3d)} visible "
                  f"(min={min_vis}), retrying...")
            precheck_ok = False
            break

        all_kps_2d[tgt_name] = (kp_names_t, kps_2d, kps_3d)

    if not precheck_ok:
        for op in pre_render_ops:
            op.cleanup(ctx)
        continue

    # ── 渲染 ──
    img_name = f"{idx:05d}.png"
    img_path = os.path.join(args.output_dir, "images", img_name)
    ctx.img_path = img_path
    scene.render.filepath = img_path
    scene.frame_current = idx
    bpy.ops.render.render(write_still=True)

    # ── 渲染后同步 ──
    ctx.K = get_camera_K(cam_data, scene)
    ctx.R_w2c, ctx.t_w2c = blender_cam_to_opencv(cam_obj)
    ctx.cam_pos = (-ctx.R_w2c.T @ ctx.t_w2c)

    # ── segmap → bbox ──
    seg_frame_str = str(idx).zfill(4)
    targets_ann = _process_segmaps(
        seg_frame_str, _target_seg_nodes, _targets_cfg,
        _segmap_dir, ctx.render_w, ctx.render_h)

    if targets_ann is None:
        print(f"[render] frame {idx}: segmap bbox missing for some target, "
              f"discarding frame and retrying...")
        if os.path.exists(img_path):
            os.remove(img_path)
        for op in pre_render_ops + post_render_ops:
            op.cleanup(ctx)
        continue

    # ── 每个 target 写入关键点标注 ──
    primary_kps_2d = None
    primary_kp_names = None
    primary_kp_3d = None

    for tgt_name, (kp_names_t, kps_2d, kps_3d) in all_kps_2d.items():
        # 渲染后重投影 (保留 pre-check 的 vis)
        old_vis = [kp[2] for kp in kps_2d]
        kps_2d_new = project_keypoints(
            kps_3d, ctx.K, ctx.R_w2c, ctx.t_w2c, ctx.render_w, ctx.render_h)
        for kp, v in zip(kps_2d_new, old_vis):
            kp[2] = v

        if tgt_name in targets_ann:
            targets_ann[tgt_name]["keypoints_2d"] = kps_2d_new
            targets_ann[tgt_name]["keypoint_names"] = kp_names_t

        if primary_kps_2d is None:
            primary_kps_2d = kps_2d_new
            primary_kp_names = kp_names_t
            primary_kp_3d = kps_3d

    # ── 组装标注 ──
    ann = {
        "id":             idx,
        "file_name":      img_name,
        "width":          ctx.render_w,
        "height":         ctx.render_h,
        "keypoint_names": primary_kp_names or [],
        "keypoints_2d":   primary_kps_2d or [],
        "keypoints_3d":   [p.tolist() for p in primary_kp_3d] if primary_kp_3d else [],
        "targets":        targets_ann,
        "bbox_2d":          targets_ann["pillar"]["bbox_2d"],
        "bbox_2d_exchange": targets_ann["exchange"]["bbox_2d"],
        "camera_K":       ctx.K.tolist(),
        "camera_R":       ctx.R_w2c.tolist(),
        "camera_t":       ctx.t_w2c.tolist(),
        "camera_pos_world": ctx.cam_pos.tolist(),
        "meta": {
            **ctx.meta,
            "light_type":      light_type,
            "n_visible":       sum(1 for kp in (primary_kps_2d or []) if kp[2] == 2),
            "n_occluded":      sum(1 for kp in (primary_kps_2d or []) if kp[2] == 1),
        },
    }
    annotations.append(ann)

    for op in pre_render_ops + post_render_ops:
        op.cleanup(ctx)

    idx += 1
    if (idx - start_idx) % 10 == 0:
        print(f"[render] {idx - start_idx}/{n_this_shard} done")

# ─────────────────────────────────────────────────
#  保存标注
# ─────────────────────────────────────────────────
shard_suffix = f"_shard{args.shard_id}" if args.n_shards > 1 else ""
_primary_tgt = next((t for t in _targets_cfg if t.get("keypoints")), None)
output = {
    "targets": [
        {
            "name":           tgt["name"],
            "pass_index":     tgt["pass_index"],
            "keypoint_names": list(tgt.get("keypoints", {}).keys()),
            "keypoints_3d":   list(tgt.get("keypoints", {}).values()),
        }
        for tgt in _targets_cfg
    ],
    "keypoint_names":    primary_kp_names or [],
    "keypoint_3d_world": list(_primary_tgt["keypoints"].values()) if _primary_tgt else [],
    "images":            annotations,
}
ann_path = os.path.join(args.output_dir, f"annotations{shard_suffix}.json")
with open(ann_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n[render] 完成! {len(annotations)}/{n_this_shard} 张, 标注: {ann_path}")
