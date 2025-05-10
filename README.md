## Env Setup

Downlaod Repository
```bash
git clone git@github.com:tars3017/Co-DETR-MVA.git
```

### Setup conda env
OS: Ubuntu 22.04.2 LTS \
Recommend CUDA version: 11.3 \
Single GPU Memory (Inferece) `>=5GB`, but it will be slow.
Recommend with memory `24GB`

```bash
conda create -n codetr python=3.7.11 -c pytorch -c conda-forge
conda activate codetr
```

Install `rust`
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Install python packages
```bash
cd Co-DETR-MVA
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

## Checkpoint Download
[Google Drive](https://drive.google.com/drive/folders/1i-LlXWHwdfIXPt7ICyhxUeD0L3R9EFy_?usp=drive_link) \
Please place it at `models/full_cropped1_val1.pth`

Command:
```
cd models
gdown <file_id>
```
> If `gdown` fails, maybe could use `rclone` or download manually.


## Validating Detection Result
```bash
CUDA_VISIBLE_DEVICES=x bash tools/dist_test.sh  cropped/co_dino_5scale_vit_large_coco.py <checkpoint_path> <gpu_count> --eval bbox
```

## **Testing Script (Tracking Inference)**
- Adjust `batch_size` to meet gpu memory limit
- Can add `--reverse` to inverse from the opposite sequence, so as to fasten the inference process
- `<folder_to_inference>` should have same structure as `pub_test`
e.g.
```
datasets/SMOT4SB/private_test
├── 0001
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── 00003.jpg
│   ├── 00004.jpg
│   ├── 00005.jpg
│   ├── 00006.jpg
│   ├── 00007.jpg
│   ├── 00008.jpg
...
```
- **Command**
(Under main project folder, i.e. `Co-DETR-MVA/`)
```bash
CUDA_VISIBLE_DEVICES=x python tracking_outputs/vit_val.py --batch <batch_size> --folder <folder_to_inference> [--reverse] --output <output_folder>
```

Example:
```bash
CUDA_VISIBLE_DEVICES=0 python tracking_outputs/vit_val.py --batch 16 --folder datasets/SMOT4SB/pub_test --output tracking_outputs
CUDA_VISIBLE_DEVICES=1 python tracking_outputs/vit_val.py --batch 16 --folder datasets/SMOT4SB/pub_test --reverse --output tracking_outputs
```
Then these commands will place the outputs `txt` file under `tracking_outputs/pub_test`

## Acknowledgement
This repository is mostly modified from [Co-DETR](https://github.com/Sense-X/Co-DETR.git), and include some of the code from [MVA2025-SMOT4SB](https://github.com/IIM-TTIJ/MVA2025-SMOT4SB), [sahi](https://github.com/obss/sahi), [boxmot](https://github.com/mikel-brostrom/boxmot), and work from MVA2023-SOD Challenge.