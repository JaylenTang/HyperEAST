# HyperEAST: An Enhanced Attention-Based Spectral–Spatial Transformer with Self-Supervised Pretraining for Hyperspectral Image Classification
The official repository of the paper [**HyperEAST: An Enhanced Attention-Based Spectral–Spatial Transformer with Self-Supervised Pretraining for Hyperspectral Image Classification**](https://ieeexplore.ieee.org/document/11129658),  
published at *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS)*, 2025, Art. no. 11129658, doi: [10.1109/JSTARS.2025.11129658](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11129658).


<img width="5139" height="2905" alt="HyperEAST" src="https://github.com/user-attachments/assets/3e584b28-6237-40bf-822f-1780fccc3e22" />

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

```bash
pip install -r requirements.txt
```
Download datasets and pretrained checkpoints












# Citation

```bibtex
@ARTICLE{11129658,
  author={Tang, Jialin and Ma, Nan and Jia, Chen and Tian, Rui and Guo, Yanhui},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={HyperEAST: An Enhanced Attention-Based Spectral-Spatial Transformer with Self-Supervised Pretraining for Hyperspectral Image Classification}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  keywords={Transformers;Hyperspectral imaging;Computational modeling;Computer architecture;Complexity theory;Attention mechanisms;Image reconstruction;Feature extraction;Adaptation models;Context modeling;Hyperspectral image classification;Vision Transformer;self-supervised learning;linear fusion attention},
  doi={10.1109/JSTARS.2025.3599855}}
```
🙏 Acknowledgement

This repository builds upon the following works:

Factoformer
CAS-ViT
```


![HyperEAST](https://github.com/user-attachments/assets/a62cee86-0f3c-40e9-9408-ab7edc387eb1)

![LFAM](https://github.com/user-attachments/assets/dcd38ee5-116f-4d07-9842-86d6c292e1c3)
```



