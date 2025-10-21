# HyperEAST: An Enhanced Attention-Based Spectral–Spatial Transformer with Self-Supervised Pretraining for Hyperspectral Image Classification
The official repository of the paper [**HyperEAST: An Enhanced Attention-Based Spectral–Spatial Transformer with Self-Supervised Pretraining for Hyperspectral Image Classification**](https://ieeexplore.ieee.org/document/11129658),  
published at *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS)*, 2025, Art. no. 11129658, doi: [10.1109/JSTARS.2025.11129658](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11129658).

<img width="5139" height="2905" alt="HyperEAST" src="https://github.com/user-attachments/assets/3e584b28-6237-40bf-822f-1780fccc3e22" />
<img width="9770" height="2870" alt="LFAM" src="https://github.com/user-attachments/assets/bee171e5-1cf1-4208-9dd5-4330f9c828d0" />

## 📰 News
- **[2025-09]** Pretraining code released  
- **[2025-08]** Finetuning and testing code released with pretrained models.

## 🧩 Usage

### Set up the environment and install required packages

- Create [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment with Python:

```bash
conda create --name hypereast python=3.7
conda activate hypereast
```
- Install PyTorch with suitable cudatoolkit version. See [here](https://pytorch.org/get-started/locally/):

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
- Install other requirements:
```bash
pip install -r requirements.txt
```
### 📦 Download datasets and pretrained checkpoints

- Download **Indian Pines**, **Pavia University**, and **Houston** datasets using the link provided in [SpectralFormer](https://github.com/danfenghong/SpectralFormer).  
- Download **Wuhan (WHU-HI)** datasets with `.mat` file format from [here](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer).  
  *(Download the split with 100 samples per class)*  
- Download our **pretrained** and **finetuned** checkpoints from the links provided in the following table.


| Dataset | Overall Acc. (%) | Average Acc. (%) | Kappa (%) | CE Loss Ratio | FL Loss Ratio | Pretrained Model | Finetuned Model |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| IP | **93.81** | 96.89 | 92.91 | 0.5 | 0.5 | [spectral_ip](https://github.com/JaylenTang/HyperEAST/blob/main/data/IndianPine/pretrained_spectral.pth) \| [spatial_ip](https://github.com/JaylenTang/HyperEAST/blob/main/data/IndianPine/pretrained_spatial.pth) | [finetuned_ip](https://github.com/JaylenTang/HyperEAST/blob/main/finetuned_ip.pt) |
| HC | **93.00** | 91.82 | 91.83 | 0.2 | 0.8 | [spectral_hc](https://github.com/JaylenTang/HyperEAST/blob/main/data/WHU-Hi-HC/pretrained_spectral.pth) \| [spatial_hc](https://github.com/JaylenTang/HyperEAST/blob/main/data/WHU-Hi-HC/pretrained_spatial.pth) | [finetuned_hc](https://github.com/JaylenTang/HyperEAST/blob/main/finetuned_hc.pt) |
| LK | **98.87** | 99.75 | 98.51 | 0.8 | 0.2 | [spectral_lk](https://github.com/JaylenTang/HyperEAST/blob/main/data/WHU-Hi-LK/pretrained_spectral.pth) \| [spatial_lk](https://github.com/JaylenTang/HyperEAST/blob/main/data/WHU-Hi-LK/pretrained_spatial.pth) | [finetuned_lk](https://github.com/JaylenTang/HyperEAST/blob/main/finetuned_lk.pt) |
| PU | **95.36** | 95.24 | 93.76 | 0.6 | 0.4 | [spectral_pu](https://github.com/JaylenTang/HyperEAST/blob/main/data/University%20of%20Pavia/pretrained_spectral.pth) \| [spatial_pu](https://github.com/JaylenTang/HyperEAST/blob/main/data/University%20of%20Pavia/pretrained_spatial.pth) | [finetuned_pu](https://github.com/JaylenTang/HyperEAST/blob/main/finetuned_pu.pt) |
> 📁 *All checkpoints and datasets should be placed under the project root directory.*

| Dataset | Overall Acc. (%) | Average Acc. (%) | Kappa (%) |
|:--:|:--:|:--:|:--:|
| IP | **87.66** | 91.72 | 85.93 |
| HC | **92.17** | 91.07 | 90.87 |
| LK | **98.19** | 98.71 | 97.63 |
| PU | **95.15** | 93.72 | 93.46 |




## 🧠 Finetuning

To fine-tune **HyperEAST** using pretrained spectral and spatial models, run the following commands:

```bash
# Indian Pines
python main_finetune.py --dataset 'Indian' --epochs 80 --learning_rate 3e-4 \
--pretrained_spectral './data/pretrained_spectral.pth' \
--pretrained_spatial './data/pretrained_spatial.pth' \
--output_dir './output'

# University of Pavia
python main_finetune.py --dataset 'Pavia' --epochs 80 --learning_rate 1e-3 \
--pretrained_spectral './data/pretrained_spectral.pth' \
--pretrained_spatial './data/pretrained_spatial.pth' \
--output_dir './output'

# WHU-Hi-HanChuan / WHU-Hi-HongHu / WHU-Hi-LongKou
python main_finetune.py --dataset 'WHU-Hi-HC' --epochs 40 --learning_rate 1e-3 \
--pretrained_spectral './data/pretrained_spectral.pth' \
--pretrained_spatial './data/pretrained_spatial.pth' \
--output_dir './output'

python main_finetune.py --dataset 'WHU-Hi-LK' --epochs 40 --learning_rate 1e-3 \
--pretrained_spectral './data/pretrained_spectral.pth' \
--pretrained_spatial './data/pretrained_spatial.pth' \
--output_dir './output'


# Citation

```bibtex
@ARTICLE{HyperEAST,
  author={Tang, Jialin and Ma, Nan and Jia, Chen and Tian, Rui and Guo, Yanhui},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={HyperEAST: An Enhanced Attention-Based Spectral-Spatial Transformer with Self-Supervised Pretraining for Hyperspectral Image Classification}, 
  year={2025},
  volume={18},
  number={},
  pages={1-15},
  doi={10.1109/JSTARS.2025.3599855}}
```
##  Acknowledgement

This repository builds upon the following works:


- [**FactoFormer: Factorized Hyperspectral Transformers with Self-Supervised Pretraining**](https://ieeexplore.ieee.org/document/10360846) [[Code]](https://github.com/csiro-robotics/FactoFormer)
- [**CAS-ViT: Convolutional Additive Self-Attention Vision Transformers for Efficient Mobile Applications**](https://arxiv.org/abs/2408.03703) [[Code]](https://github.com/Tianfang-Zhang/CAS-ViT)
- [**Swin-MSP: A Shifted Windows Masked Spectral Pretraining Model for Hyperspectral Image Classification**](https://ieeexplore.ieee.org/document/10606196) [[Code]](https://github.com/teaRRe/Swin-MSP)
- [**Spectralformer: Rethinking hyperspectral image classification with transformers**](https://ieeexplore.ieee.org/document/9627165) [[Code]](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer)
- [**Masked Auto-Encoding Spectral–Spatial Transformer for Hyperspectral Image Classification**](https://ieeexplore.ieee.org/document/9931741) [[Code]](https://github.com/ibanezfd/MAEST)
- [**DeepHyperX [Code]**](https://github.com/xiachangxue/DeepHyperX)



