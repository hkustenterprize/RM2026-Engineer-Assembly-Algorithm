# PYTHONPATH=/data/datasets/zguobd/mmpose:$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=0 \
# /data/datasets/zguobd/miniconda3/envs/mm/bin/python \
# /data/datasets/zguobd/mmpose/tools/train.py \
# /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/td-hm_litehrnet30_pillar.py \
# --work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/litehrnet30_v1.0_SGD_128x128 \

# PYTHONPATH=/data/datasets/zguobd/mmpose:$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# /data/datasets/zguobd/miniconda3/envs/mm/bin/python \
# -m torch.distributed.launch --nproc_per_node=4 \
# /data/datasets/zguobd/mmpose/tools/train.py \
# /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/rtmpose_s_cspnet.py \
# --launcher pytorch \
# --work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/litehrnet30_rtmpose_v1.0_Adam_input_256x256_mlp_64

# PYTHONPATH=/data/datasets/zguobd/mmpose:$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=0 \
# /data/datasets/zguobd/miniconda3/envs/mm/bin/python \
# -m torch.distributed.launch --nproc_per_node=4 --master_port=29501 \
# /data/datasets/zguobd/mmpose/tools/train.py \
# /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/td-hm_litehrnet30_pillar.py \
# --launcher pytorch \
# --work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/litehrnet30_heatmap_Adam_input_256x256_output_128x128

# ── 对照组（geo loss）用不同端口 + 不同 GPU 组，可同时启动 ──────────────────
# PYTHONPATH=/data/datasets/zguobd/mmpose:$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# /data/datasets/zguobd/miniconda3/envs/mm/bin/python \
# -m torch.distributed.launch --nproc_per_node=4 --master_port=29501 \
# /data/datasets/zguobd/mmpose/tools/train.py \
# /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/td-hm_litehrnet30_pillar_geo.py \
# --launcher pytorch \
# --work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/litehrnet30_heatmap_geo_v1.0

# PYTHONPATH=/data/datasets/zguobd/mmpose:$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# /data/datasets/zguobd/miniconda3/envs/mm/bin/python \
# -m torch.distributed.launch --nproc_per_node=4 \
# /data/datasets/zguobd/mmpose/tools/train.py \
# /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/rtmpose_s_cspnet.py \
# --launcher pytorch \
# --work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/cspnext-s_rtmpose_v1.0_Adam_input_256x256_official_params


# PYTHONPATH=/data/datasets/zguobd/mmpose:$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# /data/datasets/zguobd/miniconda3/envs/mm/bin/python \
# -m torch.distributed.launch --nproc_per_node=4 \
# /data/datasets/zguobd/mmpose/tools/train.py \
# /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/td-hm_litehrnet30_pillar_attn_geo.py \
# --launcher pytorch \
# --work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/litehrnet30_heatmap_geo_enabled_attn_v1.0_dlt

# PYTHONPATH=/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/train:/data/datasets/zguobd/mmpose:$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=4,5,6,7  \
# /data/datasets/zguobd/miniconda3/envs/mm/bin/python \
# -m torch.distributed.launch --nproc_per_node=4 \
# /data/datasets/zguobd/mmpose/tools/train.py \
# /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/model_configs/td-hm_litehrnet30_exchange12_v9.1.py \
# --launcher pytorch \
# --work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/litehrnet30_exchange12_v11.0

# PYTHONPATH=/data/datasets/zguobd/mmpose:$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=2,3,4,5 \
# /data/datasets/zguobd/miniconda3/envs/mm/bin/python \
# -m torch.distributed.launch --nproc_per_node=4 \
# /data/datasets/zguobd/mmpose/tools/train.py \
# /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/model_configs/td-hm_litehrnet30_pillar5.py \
# --launcher pytorch \
# --work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/litehrnet30_pillar5_v10.0_bs56

# PYTHONPATH=/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/train:/data/datasets/zguobd/mmpose:$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=4,5,6,7  \
# /data/datasets/zguobd/miniconda3/envs/mm/bin/python \
# -m torch.distributed.launch --nproc_per_node=4 \
# /data/datasets/zguobd/mmpose/tools/train.py \
# /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/model_configs/td-hm_litehrnet30_pillar5_geo_vis.py \
# --launcher pytorch \
# --work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/litehrnet30_pillar5_geo_vis_v11.0

# ── Exchange12 with input_size=384 ──────────────────────────────────────────
# Note:
#   This config is now a true 384x384 / 192x192 setup. The old 64-per-GPU
#   batch size was inherited from the earlier pseudo-384 run that still used
#   a 256x256 pipeline. On 24GB RTX 3090s that batch no longer fits.
#   Use smaller per-GPU batches via cfg-options for this launch.
# PYTHONPATH=/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/train:/data/datasets/zguobd/mmpose:$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# /data/datasets/zguobd/miniconda3/envs/mm/bin/python \
# -m torch.distributed.launch --nproc_per_node=4 --master_port=29611 \
# /data/datasets/zguobd/mmpose/tools/train.py \
# /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/model_configs/td-hm_litehrnet30_exchange12_384.py \
# --launcher pytorch \
# --work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/litehrnet30_exchange12_384_v11.3 \

# PYTHONPATH=/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/train:/data/datasets/zguobd/mmpose:$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# /data/datasets/zguobd/miniconda3/envs/mm/bin/python \
# -m torch.distributed.launch --nproc_per_node=4 \
# /data/datasets/zguobd/mmpose/tools/train.py \
# /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/model_configs/td-hm_cspnext-s_exchange12_v11.0.py \
# --launcher pytorch \
# --work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/cspnext-s_exchange12_v11.0_v2 \

# PYTHONPATH=/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/train:/data/datasets/zguobd/mmpose:$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# /data/datasets/zguobd/miniconda3/envs/mm/bin/python \
# -m torch.distributed.launch --nproc_per_node=4 \
# /data/datasets/zguobd/mmpose/tools/train.py \
# /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/model_configs/rtmo-s_exchange12_640.py \
# --launcher pytorch \
# --work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/rtmo-s_exchange12_640_v11.0 \


PYTHONPATH=/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/train:/data/datasets/zguobd/mmpose:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/data/datasets/zguobd/miniconda3/envs/mm/bin/python \
-m torch.distributed.launch --nproc_per_node=4 \
/data/datasets/zguobd/mmpose/tools/train.py \
/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/model_configs/td-hm_litehrnet18_exchange12_v11.0.py \
--launcher pytorch \
--work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/td-hm_litehrnet18_exchange12_v11.0 \
