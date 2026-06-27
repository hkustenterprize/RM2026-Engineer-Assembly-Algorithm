"""
ops.py — 固定相机 + 物体变换 管线 Ops

Op 执行顺序 (pre-render):
  FixedCameraOp → LightingSetupOp → ArcLightsOp → MainLightsOp
  → AuxLightsOp → GlareSetupOp → ObjectTransformOp

FixedCameraOp    : 一次性设定相机位姿与内参 (之后不变)
LightingSetupOp  : 一次性设定灯条材质 + 世界背景 + 太阳灯
ArcLightsOp      : 一次性设定 36 条弧形灯条 (白色统一亮/灭)
MainLightsOp     : 一次性设定主侧板灯 (Diffuse + Area light)
AuxLightsOp      : 每帧随机辅助点光源 (环境光扰动)
GlareSetupOp     : 一次性设定 Compositor Glare 节点 (灯条光晕)
ObjectTransformOp: 每帧采样 yaw/pitch/roll/distance/offset, 通过 Empty 父级
                   整体变换所有场景物体, 并做取景检查
"""
import math
import random
import numpy as np

import bpy
from mathutils import Vector, Euler, Matrix

from .utils import (
    get_camera_K, blender_cam_to_opencv,
    project_keypoints, all_kps_in_frame,
    transform_keypoints_3d,
)


def _find_emission_node(node_tree, preferred_name: str):
    """在材质节点树中查找用于调强度的 Emission 节点."""
    if node_tree is None:
        return None
    nodes = node_tree.nodes
    n = nodes.get(preferred_name)
    if n is not None and getattr(n, "type", "") == "EMISSION":
        return n
    for cand in nodes:
        if getattr(cand, "type", "") == "EMISSION":
            return cand
    return None


# ─────────────────────────────────────────────────
#  基类
# ─────────────────────────────────────────────────

class Op:
    """操作基类."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def apply(self, ctx) -> dict:
        raise NotImplementedError

    def cleanup(self, ctx):
        pass

    def skip(self, ctx) -> dict:
        return {}

    def __call__(self, ctx) -> dict:
        meta = self.apply(ctx)
        if meta:
            ctx.meta.update(meta)
        return meta


# ─────────────────────────────────────────────────
#  FixedCameraOp — 一次性设定相机
# ─────────────────────────────────────────────────

class FixedCameraOp(Op):
    """固定相机: 设定位置、朝向、FOV, 之后每帧不变.

    YAML camera 块:
      position:  [x, y, z]   相机世界坐标 (m)
      look_at:   [x, y, z]   对准点 (仅启动时用于计算朝向)
      fov_h_deg: float        水平视场角 (度)
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._done = False
        self._cached_meta = {}

    def apply(self, ctx) -> dict:
        if self._done:
            ctx.K = self._K
            ctx.R_w2c = self._R
            ctx.t_w2c = self._t
            ctx.cam_pos = self._cam_pos
            return self._cached_meta

        c = self.cfg
        cam_obj = ctx.cam_obj
        cam_data = ctx.cam_data
        scene = ctx.scene

        pos = Vector(c["position"])
        look_at = Vector(c["look_at"])
        fov_h_deg = c["fov_h_deg"]

        cam_obj.location = pos

        direction = look_at - pos
        rot_quat = direction.to_track_quat("-Z", "Y")
        cam_obj.rotation_euler = rot_quat.to_euler()

        fov_h = math.radians(fov_h_deg)
        cam_data.lens = (cam_data.sensor_width / 2.0) / math.tan(fov_h / 2.0)

        bpy.context.view_layer.update()

        self._K = get_camera_K(cam_data, scene)
        self._R, self._t = blender_cam_to_opencv(cam_obj)
        self._cam_pos = (-self._R.T @ self._t)

        ctx.K = self._K
        ctx.R_w2c = self._R
        ctx.t_w2c = self._t
        ctx.cam_pos = self._cam_pos

        # 预算相机坐标系方向向量 (物体平移用)
        cam_pos_np = np.array(pos)
        look_at_np = np.array(look_at)
        fwd = look_at_np - cam_pos_np
        fwd = fwd / np.linalg.norm(fwd)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(fwd, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        up = up / np.linalg.norm(up)

        self.cam_fwd = fwd
        self.cam_right = right
        self.cam_up = up

        self._cached_meta = {
            "camera_position": [round(v, 4) for v in pos],
            "camera_fov_h_deg": round(fov_h_deg, 2),
        }
        self._done = True
        return self._cached_meta


# ─────────────────────────────────────────────────
#  LightingSetupOp — 一次性灯光
# ─────────────────────────────────────────────────

class LightingSetupOp(Op):
    """设定全局灯条材质 + 世界背景, 并支持独立灯条逐帧随机.

    YAML lighting 块:
      light_type:        red / blue / off
      strips_strength:   全局灯条 Emission 强度
      strip_lights:      独立灯条配置, 可单独控制颜色/亮灭/强度范围
      ambient_strength:  世界背景亮度
      ambient_color:     [r, g, b] 世界背景色
    """

    def __init__(self, cfg: dict, scene_cfg: dict):
        super().__init__(cfg)
        self._scene_cfg = scene_cfg
        self._globals_done = False
        self._material_cache = {}

    def _resolve_slot_indices(self, obj, managed_mat_names):
        slot_indices = []
        for i, slot in enumerate(obj.material_slots):
            m = slot.material
            if m is None:
                continue
            if m.name in managed_mat_names or m.name.startswith("light_bar"):
                slot_indices.append(i)
        if not slot_indices and obj.material_slots:
            slot_indices = [0]
        return slot_indices

    def _resolve_strength(self, group_cfg: dict, default_strength: float) -> float:
        strength_range = group_cfg.get("strength_range")
        if isinstance(strength_range, (list, tuple)) and len(strength_range) == 2:
            lo = float(strength_range[0])
            hi = float(strength_range[1])
            if hi < lo:
                lo, hi = hi, lo
            return random.uniform(lo, hi)
        return float(group_cfg.get("strength", default_strength))

    def _resolve_light_type(self, raw_type, choices=None):
        if raw_type != "random":
            return raw_type
        choices = choices or ["red", "blue"]
        valid = [c for c in choices if c in ("red", "blue", "off")]
        return random.choice(valid or ["red", "blue"])

    def _get_assign_material(self, strip_name, light_type, strength,
                             mats, mat_map, emission_name, cache_prefix):
        base_mat_name = mat_map.get(light_type, mat_map.get("off", ""))
        base_mat = mats.get(base_mat_name)
        if base_mat is None:
            return None

        if light_type not in ("blue", "red"):
            return base_mat

        cache_key = (cache_prefix, strip_name, light_type)
        assign = self._material_cache.get(cache_key)
        if assign is None:
            assign = base_mat.copy()
            assign.name = f"_{cache_prefix}_{strip_name}_{light_type}"
            self._material_cache[cache_key] = assign

        em = _find_emission_node(assign.node_tree, emission_name)
        if em:
            sock = em.inputs.get("Strength")
            if sock is not None and hasattr(sock, "default_value"):
                sock.default_value = strength
        return assign

    def _apply_group(self, light_objects, light_type, strength,
                     mat_map, emission_name, cache_prefix):
        mats = {m.name: m for m in bpy.data.materials}
        managed_mat_names = {name for name in mat_map.values() if name}

        for strip_name in light_objects:
            obj = bpy.data.objects.get(strip_name)
            if obj is None:
                continue

            assign = self._get_assign_material(
                strip_name=strip_name,
                light_type=light_type,
                strength=strength,
                mats=mats,
                mat_map=mat_map,
                emission_name=emission_name,
                cache_prefix=cache_prefix,
            )
            if assign is None:
                continue

            slot_indices = self._resolve_slot_indices(obj, managed_mat_names)
            for i in slot_indices:
                if i < len(obj.material_slots):
                    obj.material_slots[i].material = assign
            if not obj.material_slots:
                obj.data.materials.append(assign)

    def _setup_environment(self, ctx):
        if self._globals_done:
            return

        c = self.cfg
        sc = self._scene_cfg

        world = ctx.scene.world
        if world and world.use_nodes:
            bg = world.node_tree.nodes.get("Background")
            if bg:
                ac = c.get("ambient_color", [1.0, 1.0, 1.0])
                bg.inputs["Color"].default_value = (*ac, 1.0)
                bg.inputs["Strength"].default_value = c.get("ambient_strength", 0.25)

        sun_name = sc.get("sun_object", "日光")
        sun = bpy.data.objects.get(sun_name)
        sun_energy = c.get("sun_energy", 0.0)
        if sun and sun.type == "LIGHT":
            sun.data.energy = sun_energy

        self._globals_done = True

    def apply(self, ctx) -> dict:
        c = self.cfg
        sc = self._scene_cfg
        self._setup_environment(ctx)

        global_raw_type = getattr(ctx, "light_type", None) or c.get("light_type", "red")
        global_light_type = self._resolve_light_type(
            global_raw_type, c.get("light_type_choices"))
        global_strength = float(c.get("strips_strength", 1.2))
        global_mat_map = sc.get("material_map", {})
        global_emission_name = sc.get("emission_node", "自发光")
        global_objects = sc.get("light_objects", [])

        if global_objects:
            self._apply_group(
                light_objects=global_objects,
                light_type=global_light_type,
                strength=global_strength,
                mat_map=global_mat_map,
                emission_name=global_emission_name,
                cache_prefix="global_strip",
            )

        strip_cfg = c.get("strip_lights", {})
        strip_objects = strip_cfg.get("objects", [])
        strip_raw_type = strip_cfg.get("color")
        if strip_raw_type is None:
            strip_light_type = global_light_type
        else:
            strip_light_type = self._resolve_light_type(
                strip_raw_type, strip_cfg.get("color_choices"))

        strip_strength = None
        strip_state = "disabled"
        strip_on_prob = float(strip_cfg.get("on_prob", 1.0))

        if strip_objects:
            strip_strength = self._resolve_strength(strip_cfg, global_strength)
            if strip_light_type != "off" and random.random() >= strip_on_prob:
                effective_strip_type = "off"
                strip_state = "prob_off"
            else:
                effective_strip_type = strip_light_type
                strip_state = "on" if effective_strip_type != "off" else "off"

            self._apply_group(
                light_objects=strip_objects,
                light_type=effective_strip_type,
                strength=strip_strength,
                mat_map=strip_cfg.get("material_map", global_mat_map),
                emission_name=strip_cfg.get("emission_node", global_emission_name),
                cache_prefix="independent_strip",
            )
        else:
            effective_strip_type = strip_light_type

        return {
            "light_type": global_light_type,
            "strips_strength": global_strength,
            "strip_lights_color": effective_strip_type,
            "strip_lights_strength": strip_strength,
            "strip_lights_on_prob": strip_on_prob,
            "strip_lights_state": strip_state,
        }


# ─────────────────────────────────────────────────
#  ArcLightsOp — 弧形灯条 (一次性)
# ─────────────────────────────────────────────────

class ArcLightsOp(Op):
    """背部弧形灯条控制 (自发光, 支持静态/闪烁/随机颜色).

    YAML lighting.arc_lights 块:
      objects:         list[str]    灯条网格名列表
      strength:        float        亮时 Emission 强度
      color:           str          white / red / blue / off / random
      color_choices:   list[str]    color=random 时的候选颜色
      off:             bool         true=强制全灭
      blink:           bool         true=每帧随机亮灭
      on_prob:         float        blink 时每帧整体亮起概率
      per_object:      bool         true=每根灯条独立随机亮灭
      min_on_count:    int          per_object 时最少点亮数量
      max_on_count:    int          per_object 时最多点亮数量
      base_material:   str          复制用基础材质 (须含 Emission 节点)
      off_material:    str          关灯材质名
      emission_node:   str          Emission 节点名
    """

    def __init__(self, cfg: dict, scene_cfg: dict):
        super().__init__(cfg)
        self._scene_cfg = scene_cfg
        self._tmp_mats: list = []
        self._on_mat = None
        self._off_mat = None
        self._initialized = False

    def _resolve_color(self, raw_color, choices=None):
        if raw_color != "random":
            return raw_color
        choices = choices or ["red", "blue"]
        valid = [c for c in choices if c in ("white", "red", "blue", "off")]
        return random.choice(valid or ["red", "blue"])

    def _color_rgba(self, color_name):
        if color_name == "red":
            return (1.0, 0.05, 0.02, 1.0)
        if color_name == "blue":
            return (0.05, 0.25, 1.0, 1.0)
        return (1.0, 1.0, 1.0, 1.0)

    def _set_on_material(self, mat, em_node_name, strength, color_name):
        if mat is None:
            return
        rgba = self._color_rgba(color_name)
        em = _find_emission_node(mat.node_tree, em_node_name)
        if em:
            s = em.inputs.get("Strength")
            if s is not None and hasattr(s, "default_value"):
                s.default_value = strength
            c = em.inputs.get("Color")
            if c is not None and hasattr(c, "default_value"):
                c.default_value = rgba
        diff = mat.node_tree.nodes.get("漫射 BSDF")
        if diff:
            dc = diff.inputs.get("Color")
            if dc is not None and hasattr(dc, "default_value"):
                dc.default_value = rgba

    def _build_on_material(self, base_mat_name, em_node_name, strength, color_name):
        base_mat = bpy.data.materials.get(base_mat_name)
        if base_mat is None:
            return None
        new_mat = base_mat.copy()
        new_mat.name = "_setup_arc_on"
        self._tmp_mats.append(new_mat)
        self._set_on_material(new_mat, em_node_name, strength, color_name)
        return new_mat

    def _ensure_materials(self, strength, base_mat_name, off_mat_name,
                          em_node_name, color_name):
        if self._initialized:
            return
        self._off_mat = bpy.data.materials.get(off_mat_name)
        self._on_mat = self._build_on_material(
            base_mat_name, em_node_name, strength, color_name)
        self._initialized = True

    def _sample_active_mask(self, n_objects, blink, on_prob, per_object,
                            min_on_count, max_on_count, force_off):
        if force_off or n_objects == 0:
            return [False] * n_objects, "off"
        if not blink:
            return [True] * n_objects, "on"
        if not per_object:
            is_on = random.random() < on_prob
            return [is_on] * n_objects, ("blink_on" if is_on else "blink_off")

        lo = max(0, int(min_on_count))
        hi = min(n_objects, int(max_on_count))
        if hi < lo:
            hi = lo

        # 先决定本帧是否整体熄灭, 再在亮帧中采样点亮数量.
        if random.random() >= on_prob:
            return [False] * n_objects, "blink_off"

        n_on = random.randint(lo, hi)
        active = [False] * n_objects
        if n_on > 0:
            for idx in random.sample(range(n_objects), n_on):
                active[idx] = True
        return active, "blink_mix"

    def apply(self, ctx) -> dict:
        c = self.cfg
        objects = c.get("objects", [])
        strength = c.get("strength", 2.0)
        is_off = c.get("off", False)
        blink = c.get("blink", False)
        on_prob = float(c.get("on_prob", 0.5))
        per_object = c.get("per_object", False)
        min_on_count = c.get("min_on_count", 0 if per_object else len(objects))
        max_on_count = c.get("max_on_count", len(objects))
        color = self._resolve_color(c.get("color", "white"), c.get("color_choices"))
        base_mat_name = c.get("base_material", "light_bar_red")
        off_mat_name = c.get("off_material", "light_bar_off")
        em_node_name = c.get("emission_node",
                             self._scene_cfg.get("emission_node", "自发光"))

        self._ensure_materials(strength, base_mat_name, off_mat_name,
                               em_node_name, color)
        self._set_on_material(self._on_mat, em_node_name, strength, color)
        if self._off_mat is None and self._on_mat is None:
            return {"arc_lights": "missing_material", "arc_strength": 0.0}

        active_mask, state = self._sample_active_mask(
            n_objects=len(objects),
            blink=blink,
            on_prob=on_prob,
            per_object=per_object,
            min_on_count=min_on_count,
            max_on_count=max_on_count,
            force_off=is_off or color == "off",
        )

        n_on = 0
        for obj_name, is_on in zip(objects, active_mask):
            obj = bpy.data.objects.get(obj_name)
            assign_mat = self._on_mat if is_on else self._off_mat
            if obj is None or assign_mat is None:
                continue
            if is_on:
                n_on += 1
            if obj.material_slots:
                obj.material_slots[0].material = assign_mat
            else:
                obj.data.materials.append(assign_mat)

        return {
            "arc_lights": state,
            "arc_strength": round(strength if n_on > 0 else 0, 3),
            "arc_on_count": n_on,
            "arc_total_count": len(objects),
            "arc_color": color,
        }


# ─────────────────────────────────────────────────
#  MainLightsOp — 主侧板灯 (一次性)
# ─────────────────────────────────────────────────

class MainLightsOp(Op):
    """主侧板灯: back_main_light_1/2 材质 + left_light/right_light Area 灯.

    YAML lighting.main_lights 块:
      objects:        list[str]   网格名
      area_lights:    list[str]   面光源名
      energy:         float       面光源能量 (W)
      diffuse_node:   str         Diffuse BSDF 节点名
      base_material:  str         复制用基础材质
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._done = False
        self._tmp_mats: list = []

    def apply(self, ctx) -> dict:
        if self._done:
            return {}

        c = self.cfg
        objects = c.get("objects", [])
        area_lights = c.get("area_lights", [])
        energy = c.get("energy", 80.0)
        diffuse_node_name = c.get("diffuse_node", "漫射 BSDF")
        base_mat_name = c.get("base_material", "main_red_light")
        chosen = ctx.light_type if ctx.light_type != "random" else "red"

        for light_name in area_lights:
            light_obj = bpy.data.objects.get(light_name)
            if light_obj is None:
                continue
            light_obj.data.energy = 0.0 if chosen == "off" else energy
            if chosen == "red":
                light_obj.data.color = (1.0, 0.0, 0.0)
            elif chosen == "blue":
                light_obj.data.color = (0.0, 0.0, 1.0)

        base_mat = bpy.data.materials.get(base_mat_name)
        for obj_name in objects:
            obj = bpy.data.objects.get(obj_name)
            if obj is None or base_mat is None:
                continue
            new_mat = base_mat.copy()
            new_mat.name = f"_setup_main_{obj_name}"
            self._tmp_mats.append(new_mat)
            diff = new_mat.node_tree.nodes.get(diffuse_node_name)
            if diff:
                if chosen == "red":
                    diff.inputs["Color"].default_value = (1.0, 0.0, 0.0, 1.0)
                elif chosen == "blue":
                    diff.inputs["Color"].default_value = (0.0, 0.0, 1.0, 1.0)
                else:
                    diff.inputs["Color"].default_value = (0.06, 0.06, 0.06, 1.0)
            if obj.material_slots:
                obj.material_slots[0].material = new_mat
            else:
                obj.data.materials.append(new_mat)

        self._done = True
        return {"main_lights_color": chosen, "main_lights_energy": round(energy, 2)}


# ─────────────────────────────────────────────────
#  AuxLightsOp — 辅助点光源 (每帧随机)
# ─────────────────────────────────────────────────

class AuxLightsOp(Op):
    """辅助随机点光源, 模拟环境光扰动.

    YAML lighting.aux_lights 块:
      count_range:  [lo, hi]  每帧灯数量
      dist:         float     灯到 pivot 的距离 (m)
      energy_range: [lo, hi]  灯能量范围 (W)
    """

    def __init__(self, cfg: dict, pivot):
        super().__init__(cfg)
        self._pivot = np.array(pivot, dtype=np.float64)
        self._tmp_objs: list = []

    def apply(self, ctx) -> dict:
        self.cleanup(ctx)

        c = self.cfg
        count_lo, count_hi = c.get("count_range", [0, 3])
        aux_dist = c.get("dist", 5.0)
        energy_lo, energy_hi = c.get("energy_range", [10.0, 500.0])

        n = random.randint(count_lo, count_hi)
        for _ in range(n):
            phi = math.radians(random.uniform(0, 360))
            theta = math.radians(random.uniform(10, 80))
            x = aux_dist * math.cos(theta) * math.cos(phi)
            y = aux_dist * math.cos(theta) * math.sin(phi)
            z = aux_dist * math.sin(theta)
            pos = Vector(self._pivot + np.array([x, y, z]))

            light_data = bpy.data.lights.new(name="_aux_light", type="POINT")
            light_data.energy = random.uniform(energy_lo, energy_hi)

            rand = random.random()
            if rand < 0.55:
                r, g, b = 1.0, 0.85, 0.7
            elif rand < 0.80:
                r, g, b = 0.8, 0.9, 1.0
            elif rand < 0.90:
                r, g, b = 1.0, 1.0, 1.0
            else:
                r = random.uniform(0.5, 1.0)
                g = random.uniform(0.5, 1.0)
                b = random.uniform(0.5, 1.0)
            light_data.color = (r, g, b)

            light_obj = bpy.data.objects.new(name="_aux_light", object_data=light_data)
            ctx.scene.collection.objects.link(light_obj)
            light_obj.location = pos
            self._tmp_objs.append(light_obj)

        return {"aux_lights_count": n}

    def cleanup(self, ctx):
        for obj in self._tmp_objs:
            bpy.data.objects.remove(obj, do_unlink=True)
        self._tmp_objs = []


# ─────────────────────────────────────────────────
#  GlareSetupOp — Compositor Glare
# ─────────────────────────────────────────────────

class GlareSetupOp(Op):
    """在 Compositor 中插入 Glare 节点 (FOG_GLOW), 模拟灯条光晕.

    YAML lighting.glare 块:
      glare_type:       str          Glare 类型 (默认 FOG_GLOW)
      threshold:        float        亮度阈值
      size:             int          扩散半径
      size_range:       [int, int]   每帧随机扩散半径范围
      mix:              float        混合比例 (-1=原图, 0=50/50, 1=全辉光)
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._done = False
        self._glare_node = None

    def _sample_size(self, cfg: dict) -> int | None:
        size_range = cfg.get("size_range")
        if isinstance(size_range, (list, tuple)) and len(size_range) == 2:
            try:
                lo = int(float(size_range[0]))
                hi = int(float(size_range[1]))
            except (TypeError, ValueError):
                return None
            if hi < lo:
                lo, hi = hi, lo
            return random.randint(lo, hi)

        try:
            return int(float(cfg.get("size", 4)))
        except (TypeError, ValueError):
            return None

    def _apply_dynamic_values(self, cfg: dict) -> dict:
        if self._glare_node is None:
            return {}

        sampled_size = self._sample_size(cfg)
        if sampled_size is not None:
            self._glare_node.size = sampled_size

        return {
            "glare": True,
            "glare_type": self._glare_node.glare_type,
            "glare_size": sampled_size,
        }

    def apply(self, ctx) -> dict:
        c = self.cfg
        if not c:
            self._done = True
            return {}

        if self._done:
            return self._apply_dynamic_values(c)

        scene = ctx.scene
        scene.use_nodes = True
        tree = scene.node_tree
        nodes = tree.nodes
        links = tree.links

        composite = next((n for n in nodes if n.type == "COMPOSITE"), None)
        if composite is None:
            self._done = True
            return {}

        incoming = [
            lnk
            for lnk in list(links)
            if lnk.to_node == composite
            and getattr(lnk.to_socket, "identifier", "") == "Image"
        ]
        cur_socket = None
        for lnk in incoming:
            if cur_socket is None:
                cur_socket = lnk.from_socket
            links.remove(lnk)
        if cur_socket is None:
            rl = next((n for n in nodes if n.type == "R_LAYERS"), None)
            if rl is None:
                self._done = True
                return {}
            cur_socket = rl.outputs["Image"]

        gl = nodes.new("CompositorNodeGlare")
        gl.name = "_pp_glare"
        gl.glare_type = c.get("glare_type", "FOG_GLOW")
        self._glare_node = gl
        try:
            gl.quality = "HIGH"
        except Exception:
            pass
        try:
            gl.threshold = float(c.get("threshold", 0.0))
        except (TypeError, ValueError):
            gl.threshold = 0.0
        try:
            gl.mix = float(c.get("mix", 0.5))
        except (TypeError, ValueError):
            pass

        links.new(cur_socket, gl.inputs["Image"])
        links.new(gl.outputs["Image"], composite.inputs["Image"])

        self._done = True
        return self._apply_dynamic_values(c)


# ─────────────────────────────────────────────────
#  ObjectTransformOp — 每帧核心
# ─────────────────────────────────────────────────

class ObjectTransformOp(Op):
    """每帧通过 Empty 父级整体变换场景物体.

    将所有 MESH 对象 parent 到一个 Empty, 每帧采样
    yaw/pitch/roll/distance/offset_x/offset_y, 设置 Empty
    的 location 和 rotation_euler, 实现物体 6-DOF 变换.

    pillar_slide (可选): 装配柱沿局部 Y 轴上下滑动, 仅移动 pillar_* 对象,
    关键点同步偏移. skip_prob 概率保持原位.

    YAML object_transform 块:
      pivot:           [x,y,z]  旋转中心
      yaw_range:       [lo,hi]  绕 Z 轴左右转 (度)
      pitch_range:     [lo,hi]  绕 X 轴俯仰 (度)
      roll_range:      [lo,hi]  绕 Y 轴倾斜 (度)
      distance_range:  [lo,hi]  物体中心到相机距离 (m)
      offset_x_range:  [lo,hi]  画面水平偏移 (m)
      offset_y_range:  [lo,hi]  画面垂直偏移 (m)
      max_retries:     int      取景重试上限
      framing_margin:  int      关键点距画边至少 px
      pillar_slide:             (可选) 装配柱上下滑动
        objects:       list[str]  glob 模式 (如 "pillar_*")
        target:        str        对应 target 名 (如 "pillar")
        skip_prob:     float      保持原位概率
        y_range:       [lo, hi]   Y 轴滑动范围 (m)
    """

    def __init__(self, cfg: dict, scene_cfg: dict, targets_cfg: list,
                 cam_op: 'FixedCameraOp'):
        super().__init__(cfg)
        self._scene_cfg = scene_cfg
        self._cam_op = cam_op
        self._empty = None
        self._pivot = np.array(cfg["pivot"], dtype=np.float64)
        # 兜底初始化，避免异常路径下未设置导致 apply() 崩溃
        self._M_init_inv = np.eye(4, dtype=np.float64)

        # 收集所有 target 的初始关键点
        self._all_kp_initial = {}
        self._all_kp_names = {}
        for tgt in targets_cfg:
            kps = tgt.get("keypoints", {})
            if kps:
                self._all_kp_names[tgt["name"]] = list(kps.keys())
                self._all_kp_initial[tgt["name"]] = [
                    np.array(v, dtype=np.float64) for v in kps.values()
                ]

        # 合并所有关键点用于取景检查
        self._framing_kps_initial = []
        for pts in self._all_kp_initial.values():
            self._framing_kps_initial.extend(pts)

        # pillar_slide 配置
        self._slide_cfg = cfg.get("pillar_slide")
        self._slide_objs = []
        self._slide_init_locs = {}
        if self._slide_cfg:
            import fnmatch as _fn
            patterns = self._slide_cfg.get("objects", ["pillar_*"])
            for obj in bpy.data.objects:
                if obj.type == "MESH" and any(_fn.fnmatch(obj.name, p) for p in patterns):
                    self._slide_objs.append(obj)
                    self._slide_init_locs[obj.name] = obj.location.copy()

    def _ensure_empty(self):
        """创建 Empty 并 parent 所有 MESH 对象 (只执行一次)."""
        if self._empty is not None:
            return

        pivot = self._pivot
        empty = bpy.data.objects.new("_obj_transform_pivot", None)
        bpy.context.scene.collection.objects.link(empty)
        empty.location = Vector(pivot)
        empty.rotation_euler = Euler((0, 0, 0))
        bpy.context.view_layer.update()
        self._M_init_inv = np.array(empty.matrix_world.inverted(), dtype=np.float64)

        inv = empty.matrix_world.inverted()
        for obj in bpy.data.objects:
            if obj.type == "MESH" and obj.parent is None:
                obj.parent = empty
                obj.matrix_parent_inverse = inv

        bpy.context.view_layer.update()
        self._empty = empty

        if self._slide_cfg:
            for obj in self._slide_objs:
                self._slide_init_locs[obj.name] = obj.location.copy()

    def _apply_pillar_slide(self, slide_y: float):
        """在 Empty 局部空间中沿 Y 轴偏移 pillar 对象."""
        for obj in self._slide_objs:
            init = self._slide_init_locs[obj.name]
            obj.location = Vector((init.x, init.y + slide_y, init.z))

    def _restore_pillar_slide(self):
        """恢复 pillar 对象到局部原位."""
        for obj in self._slide_objs:
            obj.location = self._slide_init_locs[obj.name].copy()

    def apply(self, ctx) -> dict:
        self._ensure_empty()
        empty = self._empty

        c = self.cfg
        cam = self._cam_op

        dist_lo, dist_hi = c["distance_range"]
        yaw_lo, yaw_hi = c["yaw_range"]
        pitch_lo, pitch_hi = c["pitch_range"]
        roll_lo, roll_hi = c["roll_range"]
        ox_lo, ox_hi = c["offset_x_range"]
        oy_lo, oy_hi = c["offset_y_range"]
        max_retries = c.get("max_retries", 50)
        margin = c.get("framing_margin", 30)

        # pillar slide 参数
        slide_y = 0.0
        if self._slide_cfg:
            skip_prob = self._slide_cfg.get("skip_prob", 0.0)
            if random.random() >= skip_prob:
                y_lo, y_hi = self._slide_cfg.get("y_range", [0.0, 0.1])
                slide_y = random.uniform(y_lo, y_hi)

        W, H = ctx.render_w, ctx.render_h
        K = ctx.K
        R = ctx.R_w2c
        t = ctx.t_w2c

        slide_target = self._slide_cfg.get("target", "pillar") if self._slide_cfg else None
        framing_kps = []
        for tgt_name, pts in self._all_kp_initial.items():
            for pt in pts:
                if tgt_name == slide_target and slide_y != 0.0:
                    framing_kps.append(pt + np.array([0.0, slide_y, 0.0]))
                else:
                    framing_kps.append(pt.copy())

        for _retry in range(max_retries):
            yaw = random.uniform(yaw_lo, yaw_hi)
            pitch = random.uniform(pitch_lo, pitch_hi)
            roll = random.uniform(roll_lo, roll_hi)
            dist = random.uniform(dist_lo, dist_hi)
            ox = random.uniform(ox_lo, ox_hi)
            oy = random.uniform(oy_lo, oy_hi)

            new_center = (
                self._cam_op._cam_pos
                + dist * cam.cam_fwd
                + ox * cam.cam_right
                + oy * cam.cam_up
            )

            empty.location = Vector(new_center)
            empty.rotation_euler = Euler((
                math.radians(pitch),
                math.radians(roll),
                math.radians(yaw),
            ), 'XYZ')

            if slide_y != 0.0:
                self._apply_pillar_slide(slide_y)
            else:
                self._restore_pillar_slide()

            bpy.context.view_layer.update()

            M_cur = np.array(empty.matrix_world, dtype=np.float64)
            delta_M = M_cur @ self._M_init_inv
            kp_3d_frame = transform_keypoints_3d(framing_kps, delta_M)

            if all_kps_in_frame(kp_3d_frame, K, R, t, W, H, margin):
                break
        else:
            pass

        all_kp_3d_frame = {}
        for tgt_name, kps_init in self._all_kp_initial.items():
            if tgt_name == slide_target and slide_y != 0.0:
                slid = [pt + np.array([0.0, slide_y, 0.0]) for pt in kps_init]
            else:
                slid = kps_init
            all_kp_3d_frame[tgt_name] = transform_keypoints_3d(slid, delta_M)

        ctx.yaw_deg = yaw
        ctx.pitch_deg = pitch
        ctx.roll_deg = roll
        ctx.distance_m = dist
        ctx.offset_x_m = ox
        ctx.offset_y_m = oy
        ctx.kp_3d_frame = all_kp_3d_frame
        ctx.crop_w = W
        ctx.crop_h = H
        ctx.bbox_2d = [0, 0, W - 1, H - 1]

        meta = {
            "yaw_deg": round(yaw, 2),
            "pitch_deg": round(pitch, 2),
            "roll_deg": round(roll, 2),
            "distance_m": round(dist, 4),
            "offset_x_m": round(ox, 4),
            "offset_y_m": round(oy, 4),
        }
        if self._slide_cfg:
            meta["pillar_slide_y"] = round(slide_y, 4)
        return meta

    def cleanup(self, ctx):
        self._restore_pillar_slide()


# ─────────────────────────────────────────────────
#  Pipeline builder
# ─────────────────────────────────────────────────

def build_ops(cfg: dict):
    """从完整配置构建 ops 列表.

    Returns
    -------
    (pre_render_ops, post_render_ops, scene_cfg)
    """
    scene_cfg = cfg.get("scene", {})
    targets_cfg = cfg.get("targets", [])
    light_cfg = cfg.get("lighting", {})
    pivot = cfg["object_transform"]["pivot"]

    pre_render = []

    pre_render.append(FixedCameraOp(cfg["camera"]))
    pre_render.append(LightingSetupOp(light_cfg, scene_cfg))

    if "arc_lights" in light_cfg:
        pre_render.append(ArcLightsOp(light_cfg["arc_lights"], scene_cfg))
    if "main_lights" in light_cfg:
        pre_render.append(MainLightsOp(light_cfg["main_lights"]))
    if "aux_lights" in light_cfg:
        pre_render.append(AuxLightsOp(light_cfg["aux_lights"], pivot))
    if "glare" in light_cfg:
        pre_render.append(GlareSetupOp(light_cfg["glare"]))

    cam_op = pre_render[0]  # FixedCameraOp
    pre_render.append(ObjectTransformOp(
        cfg["object_transform"], scene_cfg, targets_cfg, cam_op))

    post_render = []
    return pre_render, post_render, scene_cfg
