# Ground Penetrating Radar Image Analysis for Underground Barrier Detection by Combining YOLOv12 with Channel-wise Attention and Denoising Auto-Encoder

## 1. Overview
This repository contains the official implementation of the paper "Ground Penetrating Radar Image Analysis for Underground Barrier Detection by Combining YOLOv12 with Channel-wise Attention and Denoising Auto-Encoder".

The method integrates:

- YOLOv12: a SOTA real-time one-stage object detector

- Denoising Autoencoder: to suppress clutter and preserve hyperbolic GPR signatures

- CBAM: to highlight informative channel-spatial features

By combining denoising and attention mechanisms, the framework enhances detection performance in challenging subsurface environments where GPR B-scan images are affected by noise, clutter, and overlapping reflections.

### Key Features

- AE-enhanced preprocessing removes heterogeneous noise (soil irregularities, sensor interference), while maintaining the structural integrity of hyperbolic reflections.

- CBAM: Channel-wise + spatial attention refines multi-scale features and improves detection robustness under cluttered backgrounds.

- End-to-End Detection Pipeline: AE output consists of: original image, reconstructed image, and residual map. They are combined into 3-channel input for YOLOv12

### Dataset

The dataset is privately provided by the company, which contains GPR B-scan images. It contains 4 types of pipeline: water, sewage, rain, and gas. In this study, we focus on the gas pipeline only, which offers distinct radar signatures. 

## 2. Reproducibility



## 3. Reference

:page_with_curl: Paper

## 4. Cite

## Contributors

<a href="https://github.com/NSLab-CUK/Halal-or-Not/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=NSLab-CUK/Halal-or-Not" />
</a>

<br>

***

<a href="https://nslab-cuk.github.io/"><img src="https://github.com/NSLab-CUK/NSLab-CUK/raw/main/Logo_Dual_Wide.png"/></a>

***
