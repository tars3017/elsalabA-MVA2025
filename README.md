## Env Setup

Downlaod Repository
```bash
git clone git@github.com:tars3017/Co-DETR-MVA.git
mv Co-DETR-MVA Co-DETR
```

## Checkpoint Download
[Google Drive](https://drive.google.com/file/d/1BIGWAlvhNjMbQd6_5l7rzCXxI4oOz5AL/view?usp=sharing)
Please place it under `vit-val/`

### Setup conda env
Recommend CUDA version: 11.3
Single GPU Memory `>=10GB`

```bash
conda create -n codetr python=3.7.11 -c pytorch -c conda-forge
conda activate codetr
```

Install `rust`
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

```bash
cd Co-DETR
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install .
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install loguru filterpy scikit-learn pandas gdown ftfy regex lap shapely pybboxes yapf==0.40.1
```

<!-- ### Option 2 (Docker)
```bash
bash docker/build.sh
```
## Training
```bash
CUDA_VISIBLE_DEVICES=x bash tools/dist_train.sh <path_to_config> <gpu_counts> <folder_to_save_result>
```
Example: 
```bash
CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh vit/co_dino_5scale_vit_large_coco.py 1 vit
``` -->

## Validating Detection Result
```bash
CUDA_VISIBLE_DEVICES=x bash tools/dist_test.sh  cropped/co_dino_5scale_vit_large_coco.py cropped/epoch_1.pth 1 --eval bbox
```

## Tracking Inference
- Adjust `batch_size` to meet gpu memory limit
- Can add `--reverse` to inverse from the opposite sequence, so as to fasten the inference process
```bash
CUDA_VISIBLE_DEVICES=x python tracking_outputs/vit_val.py --batch <batch_size> --folder <folder_to_inference> [--reverse]
```

## Acknowledgement
This repository is mostly modified from [Co-DETR](https://github.com/Sense-X/Co-DETR.git), and include some of the code from [MVA2025-SMOT4SB](https://github.com/IIM-TTIJ/MVA2025-SMOT4SB), [sahi](https://github.com/obss/sahi), and [boxmot](https://github.com/mikel-brostrom/boxmot).