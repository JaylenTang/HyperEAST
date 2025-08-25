# HyperEAST: An Enhanced Attention-Based Spectral-Spatial Transformer with Self-Supervised Pretraining for Hyperspectral Image Classification
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2025.

Authors: 	Jialin Tang, Nan Ma, Chen Jia, Rui Tian, Yanhui Guo

[📄 Paper](https://ieeexplore.ieee.org/document/11129658/keywords#keywords)


# Overall

Hyperspectral images (HSIs) are essential in geoscientific applications such as resource exploration, precision agriculture, and environmental monitoring due to their rich spectral-spatial information. However, existing classification methods face notable limitations: PCA ignores spatial context, CNNs lack long-range modeling, and Vision Transformer (ViT)-based models often overfit under label-scarce conditions due to their high capacity and modality-agnostic design. To address these challenges, we propose HyperEAST, an efficient dual-branch ViT framework that explicitly decouples spectral and spatial feature modeling. At its core is a novel Linear Fusion Attention Mechanism (LFAM), which replaces dot-product attention with a softmax-free additive formulation based on lightweight convolutions, enabling local-global representation learning with linear complexity. To enhance robustness under limited labels, we adopt a modality-aware masked image modeling (MIM) strategy that separately reconstructs masked spectral and spatial tokens during self-supervised pretraining. We further introduce a dataset-aware hybrid loss combining cross-entropy and focal loss to mitigate class imbalance and sharpen decision boundaries. Experiments on four benchmark HSI datasets-WHU-Hi-HC, WHU-Hi-LK, Indian Pines, and Pavia University-demonstrate that HyperEAST achieves competitive accuracy, efficiency, and robustness. Code is available at https://github.com/JaylenTang/HyperEAST.

![HyperEAST](https://github.com/user-attachments/assets/a62cee86-0f3c-40e9-9408-ab7edc387eb1)

![LFAM](https://github.com/user-attachments/assets/dcd38ee5-116f-4d07-9842-86d6c292e1c3)


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

