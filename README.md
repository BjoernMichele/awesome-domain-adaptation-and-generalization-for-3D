[//]: # "Copyright 2024 Björn Michele"
# Awesome Domain Adaptation and Generalization for 3D

## Description
This is a collection of awesome works targeting domain adaptation and generalization for 3D data. The works are roughly categorized into Uni-Modal and Multi-Modal data.
Feel free to participate and add your latest work (by creating a pull-request or opening an issue), or to adapt the publication venue if it gets accepted.
Inspired by [awesome-domain-adaptation](https://github.com/zhaoxin94/awesome-domain-adaptation/blob/master/README.md) 


<img alt="Static Badge" src="https://img.shields.io/badge/Paper_count-161-%234285F4?style=flat&logo=googlescholar&labelColor=beige">

## Contents

[Surveys and other collections](#surveys-and-other-collections)

**Unimodal data (only 3D):** 

[Domain Adaptation](#domain-adaptation)
- [DA Classification](#da-classification)
- [DA Object Detection](#da-object-detection)
- [DA Semantic Segmentation](#da-semantic-segmentation) 

[Source-Free Domain Adaptation and Test-time Adaptation](#source-free-domain-adaptation-and-test-time-adaptation)


[Generalization](#generalization-and-robustness)

**Multi-Modal data (3D (LiDAR) and image):**

[Multi-Modal](#multi-modal)

**Others:**

[Simulation](#simulation)

[Other applications](#other-applications)




## Surveys and other collections
- A Survey on Deep Domain Adaptation for LiDAR Perception [[Arxiv]](https://arxiv.org/pdf/2106.02377.pdf) 
- Awesome Domain Adaptation in 3D [[GitHub]](https://github.com/ldkong1205/awesome-3d-da/blob/main/README.md)
- Understanding the Domain Gap in LiDAR Object Detection Networks [[Arxiv]](https://arxiv.org/pdf/2204.10024.pdf)
- Quantifying the LiDAR Sim-to-Real Domain Shift: A Detailed Investigation Using Object Detectors and Analyzing Point Clouds at Target-Level [[Arxiv]](https://arxiv.org/pdf/2303.01899.pdf)


# Unimodal (3D only)

## Domain Adaptation

### DA Classification

#### 2019

- PointDAN: A multi scale 3D Domain adaption for PointCloud Representation Network for Point Cloud Representation [[NeurIPS 2019]](https://arxiv.org/pdf/1911.02744.pdf) 

#### 2020
- A Multiclass TrAdaBoost Transfer Learning Algorithm for the Classification of Mobile LiDAR Data [[ISPRS Photo and remote sensing, 2020]](https://www.sciencedirect.com/science/article/abs/pii/S0924271620301301)
- Joint Supervised and Self-Supervised Learning for 3D Real-World Challenges [[ICPR 2020]](https://arxiv.org/pdf/2004.07392.pdf)

#### 2021

- Self-supervised Learning for Domain Adaptation on Point Clouds [[WACV 2021]](https://arxiv.org/pdf/2003.12641.pdf)
- Geometry-Aware Self-Training for Unsupervised Domain Adaptation on Object Point Clouds [[ICCV 2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zou_Geometry-Aware_Self-Training_for_Unsupervised_Domain_Adaptation_on_Object_Point_Clouds_ICCV_2021_paper.pdf) 
- RefRec: Pseudo-labels Refinement via Shape Reconstruction for Unsupervised 3D Domain Adaptation [[3DV 2021]](https://arxiv.org/pdf/2110.11036.pdf)
- A Learnable Self-supervised Task for Unsupervised Domain Adaptation on Point Clouds [[Arxiv]](https://arxiv.org/pdf/2104.05164.pdf)
- Generation For Adaption: A GAN-Based Approach for Unsupervised Domain Adaption with 3D Point Cloud Data [[Arxiv]](https://arxiv.org/pdf/2102.07373.pdf)

#### 2022
- Self-Distillation for Unsupervised 3D Domain Adaptation [[WACV 2022]](https://arxiv.org/pdf/2210.08226.pdf)
- Domain Adaptation on Point Clouds via Geometry-Aware Implicits [[CVPR 2022]](https://arxiv.org/pdf/2112.09343.pdf)
- Self-Supervised Global-Local Structure Modeling for Point Cloud Domain Adaptation with Reliable Voted Pseudo Labels [[CVPR 2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Fan_Self-Supervised_Global-Local_Structure_Modeling_for_Point_Cloud_Domain_Adaptation_With_CVPR_2022_paper.pdf)



#### 2023
- MEnsA: Mix-up Ensemble Average for Unsupervised Multi Target Domain Adaptation on 3D Point Clouds [[CVPRw 2023]](https://arxiv.org/pdf/2304.01554.pdf)
- Progressive Target-Styled Feature Augmentation for Unsupervised Domain Adaptation on Point Clouds [[Arxiv]](https://arxiv.org/pdf/2311.16474.pdf)

#### 2024
- Progressive Classifier and Feature Extractor Adaptation for Unsupervised Domain Adaptation on Point Clouds [[ECCV 2024]](https://arxiv.org/pdf/2311.16474)
- Finetuning Pre-trained Model with Limited Data for LiDAR-based 3D Object Detection by Bridging Domain Gaps [[Arxiv]](https://arxiv.org/pdf/2410.01319)

#### 2025
- Locating and Mitigating Gradient Conflicts in Point Cloud Domain Adaptation via Saliency Map Skewness [[Arxiv]](https://arxiv.org/pdf/2504.15796)

### DA Object Detection

#### 2019
- LiDAR Sensor modeling and Data augmentation with GANs for Autonomous driving [[Arxiv]](https://arxiv.org/pdf/1905.07290.pdf)
- Domain adaptation for vehicle detection from bird’s eye view LiDar Point cloud data [[Arxiv]](https://arxiv.org/pdf/1905.08955.pdf)
- Unsupervised Neural Sensor Models for Synthetic LiDAR Data Augmentation [[Arxiv]](https://arxiv.org/pdf/1911.10575.pdf) 
- Range Adaptation for 3D Object Detection in LiDAR [[Arxiv]](https://arxiv.org/pdf/1909.12249.pdf)  
- Cross-Sensor deep domain adaptation through self supervision [[Paper]](https://repository.tudelft.nl/islandora/object/uuid%3A618db181-4d0d-4384-a9b2-4c0a8925da4f/datastream/OBJ/)

#### 2020

- Train in Germany, Test in The USA: Making 3D Object Detectors Generalize [[CVPR 2020]](https://arxiv.org/pdf/2005.08139.pdf)

#### 2021

- SRDAN: Scale-aware and Range-aware Domain Adaptation Network for Cross-dataset 3D Object Detection [[CVPR 2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_SRDAN_Scale-Aware_and_Range-Aware_Domain_Adaptation_Network_for_Cross-Dataset_3D_CVPR_2021_paper.pdf)
- ST3D: Self-training for Unsupervised Domain Adaptation on 3D Object Detection [[CVPR 2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_ST3D_Self-Training_for_Unsupervised_Domain_Adaptation_on_3D_Object_Detection_CVPR_2021_paper.pdf)
- SPG: Unsupervised Domain Adaptation for 3D Object Detection via Semantic Point Generation [[ICCV 2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_SPG_Unsupervised_Domain_Adaptation_for_3D_Object_Detection_via_Semantic_ICCV_2021_paper.pdf)
- Unsupervised Domain Adaptive 3D Detection with Multi-Level Consistency [[ICCV 2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Luo_Unsupervised_Domain_Adaptive_3D_Detection_With_Multi-Level_Consistency_ICCV_2021_paper.pdf)
- Learning Transferable Features for Point Cloud Detection via 3D Contrastive Co-training [[NeurIPS 2021]](https://proceedings.neurips.cc/paper/2021/file/b3b25a26a0828ea5d48d8f8aa0d6f9af-Paper.pdf)
- Unsupervised Subcategory Domain Adaptive Network for 3D Object Detection in LiDAR [[Electronics 2021]](https://www.mdpi.com/2079-9292/10/8/927/pdf)
- Pseudo-labeling for Scalable 3D Object Detection [[Arxiv]](https://arxiv.org/pdf/2103.02093.pdf)
- Exploiting Playbacks in Unsupervised Domain Adaptation for 3D Object Detection [[Arxiv]](https://arxiv.org/pdf/2103.14198.pdf) 
- ST3D++: Denoised Self-training for Unsupervised Domain Adaptation on 3D Object Detection [[Arxiv]](https://arxiv.org/pdf/2108.06682.pdf)
- Uncertainty-aware Mean Teacher for Source-free Unsupervised Domain Adaptive 3D Object Detection [[Arxiv]](https://arxiv.org/pdf/2109.14651.pdf)
- Attentive Prototypes for Source-free Unsupervised Domain Adaptive 3D Object Detection [[Arxiv]](https://arxiv.org/pdf/2111.15656.pdf)
- Adversarial Training on Point Clouds for Sim-to-Real 3D Object Detection [[Paper]](http://research.engr.oregonstate.edu/rdml/sites/research.engr.oregonstate.edu.rdml/files/21-0495_03_ms.pdf)



#### 2022
- Investigating the Impact of Multi-LiDAR Placement on Object Detection for Autonomous Driving [[CVPR 2022]](https://arxiv.org/pdf/2105.00373.pdf)
- LiDAR Distillation: Bridging the Beam-Induced Domain Gap for 3D Object Detection [[ECCV 2022]](https://arxiv.org/pdf/2203.14956.pdf)
- Real-Time and Robust 3D Object Detection Within Road-Side LiDARs Using Domain Adaptation [[Arxiv]](https://arxiv.org/pdf/2204.00132.pdf)
- Towards Robust 3D Object Recognition with Dense-to-Sparse Deep Domain Adaptation [[Arxiv]](https://arxiv.org/pdf/2205.03654.pdf)
- Robust 3D Object Detection in Cold Weather Conditions [[Arxiv]](https://arxiv.org/pdf/2205.11925.pdf)
- CONTEXT-AWARE DATA AUGMENTATION FOR LIDAR 3D OBJECT DETECTION [[Arxiv]](https://arxiv.org/pdf/2211.10850.pdf)
- CL3D: Unsupervised Domain Adaptation for Cross-LiDAR 3D Detection [[Arxiv]](https://arxiv.org/pdf/2212.00244.pdf)
- SSDA3D: Semi-supervised Domain Adaptation for 3D Object Detection from Point Cloud [[Arxiv]](https://arxiv.org/pdf/2212.02845.pdf)



#### 2023
- Bi3D: Bi-domain Active Learning for Cross-domain 3D Object Detection [[CVPR 2023]](https://arxiv.org/pdf/2303.05886.pdf)
- Density-Insensitive Unsupervised Domain Adaption on 3D Object Detection [[CVPR 2023]](https://arxiv.org/pdf/2304.09446.pdf)
- Revisiting Domain-Adaptive 3D Object Detection by Reliable, Diverse and Class-balanced Pseudo-Labeling [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Revisiting_Domain-Adaptive_3D_Object_Detection_by_Reliable_Diverse_and_Class-balanced_ICCV_2023_paper.pdf)
- GPA-3D: Geometry-aware Prototype Alignment for Unsupervised Domain Adaptive 3D Object Detection from Point Clouds [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_GPA-3D_Geometry-aware_Prototype_Alignment_for_Unsupervised_Domain_Adaptive_3D_Object_ICCV_2023_paper.pdf)
- SSDA3D: Semi-supervised Domain Adaptation for 3D Object Detection from Point Cloud [[AAAI 2023]](https://arxiv.org/pdf/2212.02845.pdf)
- WLST: Weak Labels Guided Self-training for Weakly-supervised Domain Adaptation on 3D Object Detection [[Arxiv]](https://arxiv.org/pdf/2310.03821.pdf)
- Gradient-based Maximally Interfered Retrieval for Domain Incremental 3D Object Detection [[Arxiv]](https://arxiv.org/pdf/2304.14460.pdf)
- MS3D: Leveraging Multiple Detectors for Unsupervised Domain Adaptation in 3D Object Detection [[Arxiv]](https://arxiv.org/pdf/2304.02431.pdf)

#### 2024
- SOAP: Cross-sensor Domain Adaptation for 3D Object Detection Using Stationary Object Aggregation Pseudo-labelling [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Huang_SOAP_Cross-Sensor_Domain_Adaptation_for_3D_Object_Detection_Using_Stationary_WACV_2024_paper.pdf)
- Pseudo Label Refinery for Unsupervised Domain Adaptation on Cross-dataset 3D Object Detection [[CVPR 2024]](https://arxiv.org/pdf/2404.19384)
- CMD: A Cross Mechanism Domain Adaptation Dataset for 3D Object Detection [[ECCV 2024]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07443.pdf)
- UADA3D: Unsupervised Adversarial Domain Adaptation for 3D Object Detection with Sparse LiDAR and Large Domain Gaps [[IEEE RA-L]](https://arxiv.org/abs/2403.17633)
- LiT: Unifying LiDAR “Languages” with LiDAR Translator [[NeurIPS 2024]](https://openreview.net/pdf/b9f11e717fae53a1228a5b9c208bb323f8080693.pdf)
- Syn-to-Real Unsupervised Domain Adaptation for Indoor 3D Object Detection [[Arxiv]](https://arxiv.org/pdf/2406.11311)
- Semi-Supervised Domain Adaptation Using Target-Oriented Domain Augmentation for 3D Object Detection [[Arxiv]](https://arxiv.org/pdf/2406.11313)
- CTS: Sim-to-Real Unsupervised Domain Adaptation on 3D Detection [[Arxiv]](https://arxiv.org/pdf/2406.18129)
- STAL3D: Unsupervised Domain Adaptation for 3D Object Detection via Collaborating Self-Training and Adversarial Learning [[Arxiv]](https://arxiv.org/abs/2406.19362)
- DALI: Domain Adaptive LiDAR Object Detection via Distribution-level and Instance-level Pseudo Label Denoising [[Arxiv]](https://arxiv.org/pdf/2412.08806)


### DA Semantic Segmentation

#### 2019
- SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud [[ICRA 2019]](https://arxiv.org/pdf/1809.08495.pdf) 
- ePointDA: An End-to-End Simulation-to-Real Domain Adaptation Framework for LiDAR Point Cloud Segmentation [[AAAI 2021]](https://arxiv.org/pdf/2009.03456.pdf)
- Analyzing the Cross-Sensor Portability of Neural Network Architectures for LiDAR-based Semantic Labeling [[Arxiv]](https://arxiv.org/pdf/1907.02149.pdf)

#### 2020
- Domain Transfer for Semantic Segmentation of LiDAR Data using Deep Neural Networks [[IROS 2020]](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/langer2020iros.pdf)
- Unsupervised scene adaptation for semantic segmentation of urban mobile laser scanning point clouds [[ISPRS Photo and remote sensing 2020]](https://www.sciencedirect.com/science/article/abs/pii/S0924271620302744)  
- Improved Semantic Segmentation of Low-Resolution 3D Point Clouds Using Supervised Domain Adaptation [[NILES 2020]](https://ieeexplore.ieee.org/document/9257903)
- Domain Adaptation in LiDAR Semantic Segmentation by Aligning Class Distributions [[Arxiv]](https://arxiv.org/pdf/2010.12239.pdf)
- Point-Based Multilevel Domain Adaptation for Point Cloud Segmentation [[Paper]](https://www.researchgate.net/publication/346441330_Point-Based_Multilevel_Domain_Adaptation_for_Point_Cloud_Segmentation/link/616f9a59c891c4663aaa1da2/download)


#### 2021
- Complete & Label: A Domain Adaptation Approach to Semantic Segmentation of LiDAR Point Clouds [[CVPR 2021]](https://arxiv.org/pdf/2007.08488.pdf)
- Unsupervised Domain Adaptive Point Cloud Semantic Segmentation [[ACPR 2021]](https://link.springer.com/chapter/10.1007/978-3-031-02375-0_21)
- Cycle and Semantic Consistent Adversarial Domain Adaptation for Reducing Simulation-to-Real Domain Shift in LiDAR Bird’s Eye View [[ITSC 2021]](https://arxiv.org/pdf/2104.11021.pdf) 
- HYLDA: End-to-end Hybrid Learning Domain Adaptation for LiDAR Semantic Segmentation [[Arxiv]](https://arxiv.org/pdf/2201.05585.pdf)
- Unsupervised Domain Adaptation in LiDAR Semantic Segmentation with Self-Supervision and Gated Adapters [[Arxiv]](https://arxiv.org/pdf/2107.09783.pdf)
- LiDARNet: A Boundary-Aware Domain Adaptation Model for Point Cloud Semantic Segmentation [[Arxiv]](https://arxiv.org/pdf/2003.01174.pdf)





#### 2022
- Transfer Learning from Synthetic to Real LiDAR Point Cloud for Semantic Segmentation [[AAAI 2022]](https://arxiv.org/pdf/2107.05399.pdf)
- Unsupervised Domain Adaptation for Point Cloud Semantic Segmentation via Graph Matching [[IROS 2022]](https://arxiv.org/pdf/2208.04510.pdf)
- DODA: Data-oriented Sim-to-Real Domain Adaptation for 3D Indoor Semantic Segmentation [[ECCV 2022]](https://arxiv.org/pdf/2204.01599.pdf)
- CoSMix: Compositional Semantic Mix for Domain Adaptation in 3D LiDAR Segmentation [[ECCV 2022]](https://arxiv.org/pdf/2207.09778.pdf)
- Enhanced Prototypical Learning for Unsupervised Domain Adaptation in LiDAR Semantic Segmentation [[Arxiv]](https://arxiv.org/pdf/2205.11419.pdf)
- Fake it, Mix it, Segment it: Bridging the Domain Gap Between Lidar Sensors [[Arxiv]](https://arxiv.org/pdf/2212.09517.pdf)
- ADAS: A Simple Active-and-Adaptive Baseline for Cross-Domain 3D Semantic Segmentation [[Arxiv]](https://arxiv.org/pdf/2212.10390.pdf)
- Domain Adaptation in LiDAR Semantic Segmentation via Alternating Skip Connections and Hybrid Learning [[Arxiv]](https://arxiv.org/pdf/2201.05585.pdf)


#### 2023
- T–UDA: Temporal Unsupervised Domain Adaptation in Sequential Point Clouds [[IROS 2023]](https://arxiv.org/pdf/2309.08302.pdf)
- Compositional Semantic Mix for Domain Adaptation in Point Cloud Segmentation [[TPAMI 2023]](https://arxiv.org/abs/2308.14619)
- Adversarially Masking Synthetic to Mimic Real: Adaptive Noise Injection for Point Cloud Segmentation Adaptation [[CVPR 2023]](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Adversarially_Masking_Synthetic_To_Mimic_Real_Adaptive_Noise_Injection_for_CVPR_2023_paper.pdf)
- Adversarially Masking Synthetic to Mimic Real:Adaptive Noise Injection for Point Cloud Segmentation Adaptation [[CVPR 2023]](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Adversarially_Masking_Synthetic_To_Mimic_Real_Adaptive_Noise_Injection_for_CVPR_2023_paper.pdf)
- LiDAR-UDA: Self-ensembling Through Time for Unsupervised LiDAR Domain Adaptation [[ICCV 2023]](https://arxiv.org/pdf/2309.13523.pdf)
- ConDA: Unsupervised Domain Adaptation for LiDAR Segmentation via Regularized Domain Concatenation [[ICRA 2023]](https://arxiv.org/pdf/2111.15242.pdf)
- Prototype-Guided Multitask Adversarial Network for Cross-Domain LiDAR Point Clouds Semantic Segmentation [[IEEE Transactions on Geoscience and Remote Sensing 2023]](https://www.semanticscholar.org/paper/Prototype-Guided-Multitask-Adversarial-Network-for-Yuan-Cheng/065e8edaa9df6513694ccf23859fc2b828d0b35e) 

#### 2024 
- SALUDA: Surface-based Automotive Lidar Unsupervised Domain Adaptation [[3DV 2024]](https://arxiv.org/pdf/2304.03251.pdf)
- Construct to Associate: Cooperative Context Learning for Domain Adaptive Point Cloud Segmentation [[CVPR 2024]](https://openaccess.thecvf.com//content/CVPR2024/papers/Li_Construct_to_Associate_Cooperative_Context_Learning_for_Domain_Adaptive_Point_CVPR_2024_paper.pdf)
- Density-guided Translator Boosts Synthetic-to-Real Unsupervised Domain Adaptive Segmentation of 3D Point Clouds [[CVPR 2024]](https://arxiv.org/pdf/2403.18469.pdf)
- Contrastive Maximum Mean Discrepancy for Unsupervised Domain Adaptation Applied to Large Scale 3D LiDAR Semantic Segmentation [[Paper]](https://www.semanticscholar.org/paper/Contrastive-Maximum-Mean-Discrepancy-for-Domain-to-Mendili-Daniel/0ce42ccd4f85bffc7d10c92803d92f95d57b4296)
- LiOn-XA: Unsupervised Domain Adaptation via LiDAR-Only Cross-Modal Adversarial Training[[ArXiv]](https://arxiv.org/pdf/2410.15833)

#### 2025
- Robust Unsupervised Domain Adaptation for 3D Point Cloud Segmentation Under Source Adversarial Attacks [[ArXiv]](https://arxiv.org/pdf/2504.01659)
- Overlap-Aware Feature Learning for Robust Unsupervised Domain Adaptation for 3D Semantic Segmentation [[ArXiv]](https://arxiv.org/pdf/2504.01668)



## Source-Free Domain Adaptation and Test-time Adaptation

#### 2020
- SF-UDA3D: Source-Free Unsupervised Domain Adaptation for LiDAR-Based 3D Object Detection [[3DV 2020]](https://arxiv.org/pdf/2010.08243.pdf)

#### 2022
- GIPSO: Geometrically Informed Propagation for Online Adaptation in 3D LiDAR Segmentation [[ECCV 2022]](https://arxiv.org/pdf/2207.09763.pdf)

#### 2024
- Train Till You Drop: Towards Stable and Robust Source-free Unsupervised 3D Domain Adaptation [[ECCV 2024]](https://arxiv.org/pdf/2409.04409)
- HGL: Hierarchical Geometry Learning for Test-time Adaptation in 3D Point Cloud Segmentation [[ECCV 2024]](https://arxiv.org/pdf/2407.12387)
- MOS: Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection [[Arxiv]](https://arxiv.org/pdf/2406.14878)
- CloudFixer: Test-Time Adaptation for 3D Point Clouds via Diffusion-Guided Geometric Transformation [[Arxiv]](https://arxiv.org/pdf/2407.16193)
- Test-Time Adaptation of 3D Point Clouds via Denoising Diffusion Models [[Arxiv]](https://arxiv.org/pdf/2411.14495)


## Generalization and Robustness

#### 2022
- 3D-VField: Adversarial Augmentation of Point Clouds for Domain Generalization in 3D Object Detection [[CVPR 2022]](https://arxiv.org/pdf/2112.04764.pdf)
- Synthetic-to-Real Domain Generalized Semantic Segmentation for 3D Indoor Point Clouds [[Arxiv]](https://arxiv.org/pdf/2212.04668.pdf)

#### 2023
- Instant Domain Augmentation for LiDAR Semantic Segmentation [[CVPR 2023]](https://arxiv.org/pdf/2303.14378.pdf)
- Single Domain Generalization for LiDAR Semantic Segmentation [[CVPR 2023]](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_Single_Domain_Generalization_for_LiDAR_Semantic_Segmentation_CVPR_2023_paper.pdf)
- 3D Semantic Segmentation in the Wild: Learning Generalized Models for Adverse-Condition Point Clouds [[CVPR 2023]](https://openaccess.thecvf.com/content/CVPR2023/papers/Xiao_3D_Semantic_Segmentation_in_the_Wild_Learning_Generalized_Models_for_CVPR_2023_paper.pdf)
- Domain generalization of 3D semantic segmentation in autonomous driving [[ICCV 2023]](https://arxiv.org/pdf/2212.04245v2.pdf)
- Walking Your LiDOG: A Journey Through Multiple Domains for LiDAR Semantic Segmentation [[ICCV 2023]](https://arxiv.org/pdf/2304.11705.pdf)
- Robo3D: Towards Robust and Reliable 3D Perception against Corruptions [[ICCV 2023]](https://arxiv.org/pdf/2303.17597.pdf)
- 3D Adversarial Augmentations for Robust Out-of-Domain Predictions [[IJCV]](https://arxiv.org/pdf/2308.15479)
- Domain Generalization of 3D Object Detection by Density-Resampling [[Arxiv]](https://arxiv.org/pdf/2311.10845.pdf)
- Domain Generalization in LiDAR Semantic Segmentation Leveraged by Density Discriminative Feature Embedding [[Arxiv]](https://arxiv.org/pdf/2312.12098.pdf)

#### 2024 
- UniMix: Towards Domain Adaptive and Generalizable LiDAR Semantic Segmentation in Adverse Weather [[CVPR 2024]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_UniMix_Towards_Domain_Adaptive_and_Generalizable_LiDAR_Semantic_Segmentation_in_CVPR_2024_paper.pdf)
- An Empirical Study of the Generalization Ability of Lidar 3D Object Detectors to Unseen Domains [[CVPR 2024]](https://arxiv.org/pdf/2402.17562)
- Rethinking Data Augmentation for Robust LiDAR Semantic Segmentation in Adverse Weather [[ECCV 2024]](https://arxiv.org/pdf/2407.02286)
- DG-PIC: Domain Generalized Point-In-Context Learning for Point Cloud Understanding [[ECCV 2024]](https://arxiv.org/pdf/2407.08801)
- Rethinking LiDAR Domain Generalization: Single Source as Multiple Density Domains [[ECCV 2024]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03042.pdf)

#### 2025
- From One to the Power of Many: Invariance to Multi-LiDAR Perception from Single-Sensor Datasets [[AAAIw 2025]](https://arxiv.org/pdf/2409.18592)
- 3DLabelProp: Geometric-Driven Domain Generalization for LiDAR Semantic Segmentation in Autonomous Driving [[IEEE T-RO]](https://arxiv.org/pdf/2501.14605)
- Improving Generalization Ability for 3D Object Detection by Learning Sparsity-invariant Features [[Arxiv]](https://arxiv.org/pdf/2502.02322)
- An Iterative Task-Driven Framework for Resilient LiDAR Place Recognition in Adverse Weather [[Arxiv]](https://arxiv.org/pdf/2504.14806)

## Multi-Modal
#### 2020
- xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation [[CVPR 2020]](https://arxiv.org/pdf/1911.12676.pdf)

#### 2021
- Adversarial unsupervised domain adaptation for 3D semantic segmentation with multi-modal learning [[ISPRS 2021]](https://www.sciencedirect.com/science/article/abs/pii/S0924271621001131#!)
- mDALU: Multi-Source Domain Adaptation and Label Unification with Partial Datasets [[ICCV 2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Gong_mDALU_Multi-Source_Domain_Adaptation_and_Label_Unification_With_Partial_Datasets_ICCV_2021_paper.pdf)
- Sparse-to-dense Feature Matching: Intra and Inter domain Cross-modal Learning in Domain Adaptation for 3D Semantic Segmentation [[ICCV 2021]](https://arxiv.org/pdf/2107.14724.pdf)
- See Eye to Eye: A Lidar-Agnostic 3D Detection Framework for Unsupervised Multi-Target Domain Adaptation [[Arxiv]](https://arxiv.org/pdf/2111.09450.pdf)

#### 2022
- MM-TTA: Multi-Modal Test-Time Adaptation for 3D Semantic Segmentation [[CVPR 2022]](https://arxiv.org/pdf/2204.12667.pdf)
- Self-supervised Exclusive Learning for 3D Segmentation with Cross-Modal Unsupervised Domain Adaptation [[ACM 2022]](https://dl.acm.org/doi/10.1145/3503161.3547987)
- Cross-Domain and Cross-Modal Knowledge Distillation in Domain Adaptation for 3D Semantic Segmentation [[ACM 2022]](https://dl.acm.org/doi/10.1145/3503161.3547990)

#### 2023
- Cross-modal & Cross-domain Learning for Unsupervised LiDAR Semantic Segmentation [[ACM 2023]](https://arxiv.org/pdf/2308.02883.pdf)
- Cross-modal Unsupervised Domain Adaptation for 3D Semantic Segmentation via Bidirectional Fusion-then-Distillation [[ACM 2023]](https://dl.acm.org/doi/10.1145/3581783.3612013)
- Cross-Modal Contrastive Learning for Domain Adaptation in 3D Semantic Segmentation [[AAAI 2023]](https://ojs.aaai.org/index.php/AAAI/article/view/25400)
- Exploiting the Complementarity of 2D and 3D Networks to Address Domain-Shift in 3D Semantic Segmentation [[CVPR 2023]](https://arxiv.org/pdf/2304.02991.pdf)
- Mx2M: Masked Cross-Modality Modeling in Domain Adaptation for 3D Semantic Segmentation [[AAAI 2023]](https://arxiv.org/pdf/2307.04231.pdf) 



#### 2024
- CMDA: Cross-Modal and Domain Adversarial Adaptation for LiDAR-based 3D Object Detection [[AAAI 2024]](https://arxiv.org/pdf/2403.03721.pdf)
- Learning to Adapt SAM for Segmenting Cross-domain Point Clouds [[ECCV 2024]](https://arxiv.org/pdf/2310.08820)
- MoPA: Multi-Modal Prior Aided Domain Adaptation for 3D Semantic Segmentation [[ICRA 2024]](https://arxiv.org/pdf/2309.11839.pdf)
- UniDSeg: Unified Cross-Domain 3D Semantic Segmentation via Visual Foundation Models Prior [[NeurIPS 2024]](https://openreview.net/pdf/2a44ca78c6cb797dc9a91aebf697ca399484e8e1.pdf)
- Multimodal 3D Object Detection on Unseen Domains [[Arxiv]](https://arxiv.org/pdf/2404.11764.pdf)
- Visual Foundation Models Boost Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation [[Arxiv]](https://arxiv.org/pdf/2403.10001)
- TTT-KD: Test-Time Training for 3D Semantic Segmentation through Knowledge Distillation from Foundation Models [[Arxiv]](https://arxiv.org/pdf/2403.11691)
- Fusion-then-Distillation: Toward Cross-modal Positive Distillation for Domain Adaptive 3D Semantic Segmentation [[Arxiv]](https://arxiv.org/pdf/2410.19446)

#### 2025
- Exploring Modality Guidance to Enhance VFM-based Feature Fusion for UDA in 3D Semantic Segmentation [[Arxiv]](https://arxiv.org/pdf/2504.14231)

# Others

## Simulation

#### 2019
- CNN-based synthesis of realistic high-resolution LiDAR data [[IEEE Intelligent Vehicles Symposium]](https://arxiv.org/pdf/1907.00787.pdf)

#### 2021

- Learning to Drop Points for LiDAR Scan Synthesis [[Arxiv]](https://arxiv.org/pdf/2102.11952.pdf)

#### 2022
- LiDAR Snowfall Simulation for Robust 3D Object Detection [[CVPR 2022]](https://arxiv.org/pdf/2203.15118.pdf)
- SLiDE: Self-supervised LiDAR De-snowing through Reconstruction Difficulty [[ECCV 2022]](https://arxiv.org/pdf/2208.04043.pdf)
- Learning to Simulate Realistic LiDARs [[IROS 2022]](https://arxiv.org/pdf/2209.10986.pdf)
- DiffCloud: Real-to-Sim from Point Clouds with Differentiable Simulation and Rendering of Deformable Objects [[Arxiv]](https://arxiv.org/pdf/2204.03139.pdf)

#### 2023
- Towards Zero Domain Gap: A Comprehensive Study of Realistic LiDAR Simulation for Autonomy Testing [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Manivasagam_Towards_Zero_Domain_Gap_A_Comprehensive_Study_of_Realistic_LiDAR_ICCV_2023_paper.pdf)


## Other applications
- Weakly-Supervised Domain Adaptation via GAN and Mesh Model for Estimating 3D Hand Poses Interacting Objects [[CVPR 2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Baek_Weakly-Supervised_Domain_Adaptation_via_GAN_and_Mesh_Model_for_Estimating_CVPR_2020_paper.pdf)
- Project to Adapt: Domain Adaptation for Depth Completion from Noisy and Sparse Sensor Data [[ACCV 2020]](https://openaccess.thecvf.com/content/ACCV2020/papers/Lopez-Rodriguez_Project_to_Adapt_Domain_Adaptation_for_Depth_Completion_from_Noisy_ACCV_2020_paper.pdf)
- A Registration-Aided Domain Adaptation Network for 3D Point Cloud Based Place Recognition [[IROS 2021]](https://arxiv.org/pdf/2012.05018.pdf)
- PointSFDA: Source-free Domain Adaptation for Point Cloud Completion [[Arxiv]](https://arxiv.org/pdf/2503.15144)
- Unsupervised Domain Adaptation for 3D Keypoint Estimation via View Consistency [[Arxiv]](https://arxiv.org/pdf/1712.05765.pdf)
- Anatomy-guided domain adaptation for 3D in-bed human pose estimation [[Arxiv]](https://arxiv.org/pdf/2211.12193.pdf)


