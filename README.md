# MMGA
The code of our paper for MGGA. 
The current version is the initial version, and the relevant code will be improved in the future.

## Future Work
✅ Release Environment Configuration Tutorial
```
pip install -r requirement.txt
```
🔵 Simplify the code
## Abstract
Medical image registration is essential for aligning heterogeneous imaging data to ensure accurate anatomical correspondence, yet current methods struggle to simultaneously capture fine-grained local details and maintain global spatial coherence. To address this limitation, we propose MGGA, a MedSAM-Guided Geometry-Aware framework that fuses 2D and 3D features for medical image registration.These 2D local representations are then semantically aligned using a CLIP-based text encoder. In parallel, a lightweight 3D encoder captures global spatial dependencies across the volume. To bridge the gap between dimensions and scales, we introduce a Geometry-Aware 2D-3D Feature Fusion Module (GA-FFM), which adaptively aligns and fuses 2D and 3D features based on Gaussian Similarity and Cosine Similarity. Furthermore, a dual-layer regularization strategy combining explicit mask-level constraints and implicit semantic guidance reinforces anatomical plausibility and deformation smoothness.Extensive experiments on brain MRI (patient-to-patient and atlas-to-patient) and abdominal CT datasets demonstrate that MGGA consistently outperforms state-of-the-art methods in DSC, ASSD, and HD95, while preserving topological validity. These results highlight the effectiveness and generalizability of our framework in achieving accurate, robust, and anatomically consistent registration across diverse imaging modalities and tasks.The code is available at: https://github.com/goghfan/MGGA.

![pipeline](pictures/pipeline.png)

## Data We Used

OASIS-3  [`OASIS3`]:(https://sites.wustl.edu/oasisbrains/home/oasis-3/).
OASIS-3 is a retrospective compilation of data for 1378 participants that were collected across several ongoing projects through the WUSTL Knight ADRC over the course of 30years. Participants include 755 cognitively normal adults and 622 individuals at various stages of cognitive decline ranging in age from 42-95yrs. All participants were assigned a new random identifier and all dates were removed and normalized to reflect days from entry into study. The dataset contains 2842 MR sessions which include T1w, T2w, FLAIR, ASL, SWI, time of flight, resting-state BOLD, and DTI sequences. Many of the MR sessions are accompanied by volumetric segmentation files produced through FreeSurfer processing. PET imaging from different tracers, PIB, AV45, and FDG, totaling over 2157 raw imaging scans and the accompanying post-processed files from the Pet Unified Pipeline (PUP) are also available in OASIS-3. Additionally, 451 Tau PET sessions and post-processed PUP are now available for OASIS-3 subjects in a sub-project ‘OASIS-3_AV1451’.
 
IXI  [`IXI_DataSet`]: (https://brain-development.org/ixi-dataset/).
The IXI dataset is a collection of brain MRI scans obtained from healthy volunteers across three different hospitals in London, using a variety of scanner models and field strengths (including 1.5T and 3T). It includes multiple MRI modalities such as T1-weighted, T2-weighted, proton density (PD), and diffusion-weighted images, providing a rich resource for research in brain anatomy, image synthesis, and neuroimaging algorithm development.
 
MICCAI FLARE 2022  [`MICCAI FLARE 2022`]:(https://flare22.grand-challenge.org/).
The FLARE22 dataset is a collection of abdominal CT scans designed for multi-organ segmentation tasks, including organs such as the liver, spleen, pancreas, kidneys, and more. It features annotated training cases and additional unseen validation/test cases, sourced from multiple institutions and scanners to reflect real-world variability. The challenge emphasizes robustness across domains and accurate segmentation of both large and small organs under limited training data conditions.

```
@article{lamontagne2019oasis,
  title={OASIS-3: longitudinal neuroimaging, clinical, and cognitive dataset for normal aging and Alzheimer disease},
  author={LaMontagne, Pamela J and Benzinger, Tammie LS and Morris, John C and Keefe, Sarah and Hornbeck, Russ and Xiong, Chengjie and Grant, Elizabeth and Hassenstab, Jason and Moulder, Krista and Vlassenko, Andrei G and others},
  journal={medrxiv},
  pages={2019--12},
  year={2019},
  publisher={Cold Spring Harbor Laboratory Press}
}
@book{ma2023fast,
  title={Fast and Low-Resource Semi-supervised Abdominal Organ Segmentation: MICCAI 2022 Challenge, FLARE 2022, Held in Conjunction with MICCAI 2022, Singapore, September 22, 2022, Proceedings},
  author={Ma, Jun and Wang, Bo},
  volume={13816},
  year={2023},
  publisher={Springer Nature}
}
```

## DSC performance of different methods on three datasets
![pipeline](pictures/ct.png)
![pipeline](pictures/IXI.png)
![pipeline](pictures/OASIS.png)
