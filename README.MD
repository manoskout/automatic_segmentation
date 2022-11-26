# Automatic segmentation in organs-at-risk and prostate of the pelvic region

The purpose of this project is to build a robust model to delineate the organ from patients' 0.35T Magnetic Resonance Images (MRIs).
The whole implementation is based on state-of-the-art deep learning-based architectures. The dataset used for this project is a private dataset acquired from 0.35T MR-Linac.

## Abstract

**Purpose** : The MR-Linac is a recent device combining a linear accelerator with an MRI scanner. The improved soft tissue contrast of MR images is used for the optimum delineation of tumors or organs at risk (OARs) and precise treatment delivery. Automatic segmentation of OARs is a crucial feature to alleviate the time-consuming process of manual segmentation from physicians. In this work, current state-of-the-art deep learning techniques were evaluated based on 2D and 2.5D training strategies to develop a comprehensive tool for accurate segmentation of pelvic OARs dedicated to MR images from a 0.35T MR-Linac system.

**Methods** :In total, 103 cases with 0.35 T MR images of the pelvic region were investigated. Expert physicians contoured OARs (bladder, rectum, and femoral heads) and prostate. For the training of the neural network, 85 patients were randomly selected, and 18 patients were used for testing. Multiple state-of-the-art U-Net-based neural network architectures were investigated to automatically segment the prostate and OARs. The best model was compared in both 2D and 2.5D training strategies. The evaluation was performed by focusing on the results of two metrics: the Dice Similarity Coefficient (DSC) and the Hausdorff Distance (HD). Based on 2D architectures, Residual Attention U-Net (ResAttUNet) provides the best results among the other deep neural networks. Moreover, the 2.5D version of the configurated ResAttUNet achieved more accurate results. The overall DSC was 0.88±0.09 and 0.86±0.10, and overall HD was 1.78±3.02 mm and 5.90±7.58 mm for 2.5D ResAttUNet and 2D ResAttUNet, respectively.

**Conclusion**: The 2.5D Residual Attention U-Net architecture provides accurate segmentation of the prostate and the OARs without affecting the computational cost. The finalized tool will be merged with the treatment planning system for in-time automatic segmentation on the 0.35 T MR-Linac.

## Data management

## Preprocessing

## Architecture

## Post-processing

## Results
