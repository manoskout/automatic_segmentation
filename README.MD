### Automatic segmentation in Organs-at-risk of the pelvic region - Deep Learning approach

The purpose of this project is to build a robust model to delineate the organ from patients' Magnetic Resonance Images (MRIs).
The whole implementation is based on state-of-the-art deep learning-based methods. We briefly introduce some of the existed architectures selecting the optimal solution to our challenge. However, due to the small dataset, we had to implement data augmentation techniques to enhance the generalizability of our results.

### Abstract


### Introduction

The continuous rising of the deep learning (DL) techniques in the medical field provides a promising footings in this field. Computer-aided diagnosis and segmentation in radiology provide strong foundations for investigation using deep learning techniques. Amongst the anatomical locations where recent automatic segmentation algorithms have been proposed, the pelvic region remains a significant challenge due to the large intra- and inter-patient variability. In radiation oncology, the accurate countouring of OARs is inevitably necessary step for planning the optimal dose delivery to the affected organ, avoiding the toxicity delivery to surrounding healthy organs. Nonetheless, manual volume segmentation deemed a laborious and time-consuming process for the physicians, leading for delays for the radiation therapy (RT) treatment. Recent studies focused in DL-based methods, although, atlas-based and hybrid techniques remain state-of-the-art. DL is part of the Machine Learning (ML). To put it differently, deep learning levarage deep neural network architectures with multiple hidden layers between input and output. This methods learn feutures from the already segmented dataset by modeling complex non linear relationships. The most dominant architecture for automatic organ segmentation is the UNet including a vast amount of variations that have been already proposed. In this project we tested the most relavant UNet architectures, making an extended review of which one is the optimal solution for the automatic segmentation of pelvic region. It is worth also to refer that we focused both in multi-organ and organ-per-organ segmentation combare the results between these two different methods

### Methods and Materials
This study includes patient who underwent MRI.
Axial T2-weighted image were manually segmented by radiologists, which stated as ground truth. These collected images were used for training on the UNet architectures that are used, """"in a k-fold validation paradigm for neural network training and validation"""". The Dice score, a measure of overlap between the manually segmented and automatically derived segmentation, Hausdorff distance, a measure of distance between the manually and automatically segmented organ in a metric space.

### Results
The CNN trained on 75 examination of 75 patients. Due to the lack of medical sample a data augmentation process implemented. The customized 2D UNet architecturε has a Dice score of 0.90+-0.02 and Hausdorff Distance 2.05+-1.2.

### Conclusion
A deep learning CNN can automatically segment the OARs from clinical MR Images 


# MRIdian
