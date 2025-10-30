# Download Model Checkpoints
gdown --folder https://drive.google.com/drive/folders/1zFO4bgo7yvjT9rCJi_6Mt6_07wfr0CKU -O /workspace/CellViT_Custom/Checkpoints

# Preprocessing CoNSeP
Download from https://opendatalab.com/OpenDataLab/CoNSeP
## For Train
python preprocessing.py \
  --images_dir /workspace/CellViT_Custom/Dataset/CoNSeP/Original/Train/Images \
  --labels_dir /workspace/CellViT_Custom/Dataset/CoNSeP/Original/Train/Labels \
  --out_dir /workspace/CellViT_Custom/Dataset/CoNSeP/Preprocessed/Train \
  --image_suffix .png \
  --label_suffix .mat \
  --save_binary

## For Test
python preprocessing.py \
  --images_dir /workspace/CellViT_Custom/Dataset/CoNSeP/Original/Test/Images \
  --labels_dir /workspace/CellViT_Custom/Dataset/CoNSeP/Original/Test/Labels \
  --out_dir /workspace/CellViT_Custom/Dataset/CoNSeP/Preprocessed/Test \
  --image_suffix .png \
  --label_suffix .mat \
  --save_binary

# Evaluation CoNSeP on Best Model Checkpoints
python eval_instance_CoNSeP.py \
  --ckpt Checkpoints/CellViT/cellvit_vit256_consep_merged_best.pth \
  --test_root Dataset/CoNSeP/Preprocessed/Test \
  --batch_size 8 --device cuda \
  --iou_thr 0.5 --bin_thresh 0.5 --min_area 10 \
  --fg_index -1 --bg_index 0 \
  --class_names bg,epithelial,inflammatory,spindleshaped,miscellaneous

# 이게 맞는거 같은데 확인 필요.
python eval_instance_CoNSeP.py \
  --ckpt Checkpoints/CellViT/cellvit_vit256_consep_merged_best.pth \
  --test_root Dataset/CoNSeP/Preprocessed/Test \
  --batch_size 8 --device cuda \
  --iou_thr 0.5 --bin_thresh 0.5 --min_area 10 \
  --fg_index -1 --bg_index 0 \
  --class_names bg,other,inflammatory,epithelial,spindle \
  --save_viz_dir viz_out --viz_max 64

# CellViT Train at CoNSeP
python train_custom_ViT.py \
  --dataset consep_merged \
  --train_root Dataset/CoNSeP/Preprocessed/Train \
  --val_root   Dataset/CoNSeP/Preprocessed/Test \
  --num_classes 5 \
  --epochs 130 --batch_size 8 --lr 1e-4 \
  --freeze_epochs 25  \
  --init_full_ckpt Checkpoints/CellViT/CellViT-256-x40.pth

# CoNSeP Vizualization
python viz_predictions.py \
  --dataset consep_merged \
  --root Dataset/CoNSeP/Preprocessed/Test \
  --num_classes 5 \
  --ckpt Checkpoints/CellViT/cellvit_vit256_consep_merged_best.pth \
  --out eval_out/consep_merged \
  --batch_size 4 --workers 2 \
  --max_samples 40 \
  --exclude_bg

# TODO 둘 중 뭐가 좋을지 알아내기
python viz_predictions.py \
  --ckpt Checkpoints/CellViT/cellvit_vit256_consep_merged_best.pth \
  --test_root Dataset/CoNSeP/Preprocessed/Test \
  --batch_size 8 --device cuda \
  --out eval_out \
  --save_csv eval_out/per_image_metrics.csv

# Custom Image Inference pixel-wise
python infer_viz_images.py \
  --ckpt Checkpoints/CellViT/cellvit_vit256_consep_merged_best.pth \
  --input_dir Dataset/1024_crop/Preprocessed \
  --out eval_out/1024_crop \
  --batch_size 8 --device cuda \
  --img_size 256 --num_classes 5 \
  --np_thresh 0.4 --min_size 10 \
  --bnd_thickness 2 --bnd_alpha 0.95

# Custom Image Inference Panoptic
python infer_viz_images_panoptic.py \
  --ckpt Checkpoints/CellViT/cellvit_vit256_consep_merged_best.pth \
  --input_dir Dataset/1024_crop/Preprocessed \
  --out eval_out/1024_crop_panoptic \
  --batch_size 8 --device cuda \
  --img_size 256 --num_classes 5 \
  --np_thresh 0.4 --min_size 10 \
  --bnd_thickness 1 --bnd_alpha 0.95 \
  --core_erode_px 2

# PanNuke CV3 Train from Scratch at PaNNuke
python train_eval_pannuke_cv3.py \
  --hf_repo RationAI/PanNuke \
  --epochs 130 --freeze_epochs 25 \
  --batch_size 192 --lr 1e-4 \
  --out Checkpoints/CellViT/cv3

# PanNuke CV3 Train from Checkpoints
python train_custom_ViT.py   
  --hf_repo RationAI/PanNuke \
  --epochs 130 --freeze_epochs 25 \
  --batch_size 8 --lr 1e-4 \
  --init_full_ckpt Checkpoints/CellViT/CellViT-256-x40.pth \
  --out Checkpoints/CellViT/cv3


# CV Fold Evaludation with Instance
python eval_instance_pannuke_cv3.py \
  --hf_repo RationAI/PanNuke \
  --ckpt_root /workspace/CellViT_Custom/Checkpoints/CellViT/cv3 \
  --out eval_instance_cv3
