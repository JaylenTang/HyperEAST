# HyperEAST


PyTorch codes for "HyperEAST: An Enhanced Attention-Based Spectral-Spatial Transformer with Self-Supervised Pretraining for Hyperspectral Image Classification", 

Submitted to IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2025.

Authors: 	Jialin Tang, Nan Ma, Chen Jia, Rui Tian, Yanhui Guo

# Overall

Abstract—Hyperspectral images (HSIs) play a vital role in
geoscientific applications such as resource exploration, precision agriculture, and environmental monitoring, owing to their
rich spectral–spatial information. However, existing classification
methods face several limitations: traditional approaches such as
PCA overlook spatial context, CNNs struggle with long-range
dependencies, and Vision Transformer (ViT)-based models often
overfit under low-label conditions due to their high capacity and
modality-agnostic design. To address these issues, we propose
HyperEAST, an efficient dual-branch ViT-based framework that
explicitly decouples spectral and spatial feature modeling. At its
core, we introduce a novel Linear Fusion Attention Mechanism
(LFAM) that replaces conventional dot-product attention with a
softmax-free, convolution-enhanced additive formulation. LFAM
enables efficient local–global representation learning with linear
complexity, offering improved generalization and lower computation overhead. To further alleviate overfitting and enhance
spectral–spatial disentanglement in label-scarce scenarios, we
adopt a modality-aware masked image modeling (MIM) strategy that independently masks and reconstructs spectral and
spatial tokens during self-supervised pretraining. Additionally,
we design a dataset-aware hybrid loss that integrates cross entropy and focal loss to address class imbalance while improving model sensitivity to category boundaries. Experiments
on four benchmark HSI datasets—WHU-Hi-HC, WHU-Hi-LK,
Indian Pines, and Pavia University—demonstrate that HyperEAST achieves state-of-the-art classification accuracy, robustness,
and efficiency.
