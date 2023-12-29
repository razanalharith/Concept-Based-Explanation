
# <p align=center>`Concept-Based-Explanation`

Concept-based explanation methods have emerged as a major way to enhance deep learning model interpretability and transparency. By employing human-understandable concepts and explanations, these strategies try to provide insights into how these models create predictions. Concept-based explanation approaches bridge the gap between the model's internal workings and human understanding by mapping the model's internal representations to meaningful concepts such as objects or attributes.

< **Last updated: 29/12/2023** >


## 1. Content

- [List for Concept-Based-Explanation methods](#list-for-concept-based-explanation-methods)
  - [1. Content](#1-content)
  - [2. Lists](#2-lists)

- [2. Lists](#2-list)
  - [2.1. Common used Datasets Lists](#21-datasets)
    - [2.1.1 Image Datasets](#211-image-datasets)
    - [2.1.2 Video Datasets](#212-video-datasets)
    - [2.1.3 Text Datasets](#213-text-datasets)
    - [2.1.4 Chemistry Datasets](#214-chemistry-datasets)
    - [2.1.5 Biology Datasets](#215-biology-datasets)
    - [2.1.6 Graph Datasets](#216-graph-datasets)

  - [2.2. Concept-Based-Explanation methods Lists](#22-concept-based-explanation-methods)
    - [2.2.1. YEAR 2023](#221-year-2023)
    - [2.2.2. YEAR 2022](#222-year-2022)
    - [2.2.3. YEAR 2021](#223-year-2021)
    - [2.2.4. YEAR 2020](#224-year-2020)
    - [2.2.5. YEAR 2019](#225-year-2019)
    - [2.2.6. YEAR 2018](#226-year-2018)


## 2. Lists

### 2.1. Common used Datasets Lists

#### 2.1.1 Image Datasets

| No | Datasets | Year | Tasks | URL |
| -- | -------- | ---- | ----- | --- |
| 1  | ImageNet | 2009 | Classification + Detection | [Yes](http://www.image-net.org/) |
| 2  | Derm7pt  | 2019 | Detection | [Yes](https://derm.cs.sfu.ca/Welcome.html) |
| 3  | Microsoft COCO | 2014 | Detection | [Yes](https://cocodataset.org/) |
| 4  | Caltech-UCSD Birds-200-2011 (CUB) | 2011 | Classification | [Yes](https://www.vision.caltech.edu/datasets/cub_200_2011/) |
| 5  | Animal with Attributes (AwA2) | 2018 | Classification | [Yes](https://paperswithcode.com/dataset/awa2-1) |
| 6  | CIFAR-10 and CIFAR-100 | 2009 | Classification | [Yes](https://www.cs.toronto.edu/~kriz/cifar.html) |
| 7  | Places365 | 2015 | Scene Recognition | [Yes](https://paperswithcode.com/dataset/places365) |
| 8  | Skin Cancer MNIST: HAM10000 | 2018 | Detection | [Yes](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| 9  | NIH Chest X-ray | 2019 | Detection | [Yes](https://datasets.activeloop.ai/docs/ml/datasets/nih-chest-x-ray-dataset/) |
| 10 | MIT-States | 2015 | Zero Shot Learning | [Yes](http://web.mit.edu/phillipi/Public/states_and_transformations/index.html) |
| 11 | MNIST | 2010 | Classification | [Yes](http://yann.lecun.com/exdb/mnist/) |
| 12 | Myocardial infarction complications | 2020 | Classification | [Yes](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications) |
| 13 | ADE20K | 2017 | Segmentation | [Yes](https://groups.csail.mit.edu/vision/datasets/ADE20K/) |
| 14 | SIIM-ISIC | 2020 | Detection | [Yes](https://www.kaggle.com/c/siim-isic-melanoma-classification/overview) |
| 15 | CelebA | 2018 | Detection | [Yes](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| 16 | SUN attribute | 2010 | Scene Categorization | [Yes](https://vision.princeton.edu/projects/2010/SUN/) |
| 17 | IP102 | 2019 | Classification | [Yes](https://github.com/xpwu95/IP102) |
| 18 | MONK's Problems | 1992 | Classification | [Yes](https://archive.ics.uci.edu/dataset/70/monk+s+problems) |

#### 2.2.2 Video Datasets

| No | Datasets | Year | Tasks | URL |
| -- | -------- | ---- | ----- | --- |
| 1 | Kinetics-700 | 2019 | Action recognition | [Yes](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/kinetics/README.md) |
| 2 | BDD-OIA | 2020 | Scene Categorization | [Yes](https://drive.google.com/drive/folders/1NzF-UKaakHRNcyghtaWDmc-Vpem7lyQ6) |
| 3 | MSR-V2E A | 2020 | Video classification | No |

#### 2.2.3 Text Datasets

| No | Datasets | Year | Tasks | URL |
| -- | -------- | ---- | ----- | --- |
| 1 | CEBaB | 2022 | NLP | [Yes](https://cebabing.github.io/CEBaB/) |

#### 2.2.4 Chemistry Datasets

| No | Datasets | Year | Tasks | URL |
| -- | -------- | ---- | ----- | --- |
| 1 | Mutagenicity (MUTAG) | 1991 | Graph classification | [Yes](https://paperswithcode.com/sota/graph-classification-on-mutag) |

#### 2.2.5 Biology Datasets

| No | Datasets | Year | Tasks | URL |
| -- | -------- | ---- | ----- | --- |
| 1 | ENZYMES  | NA | Graph classification | [Yes](https://paperswithcode.com/dataset/enzymes) |

#### 2.2.6 Graph Datasets

| No | Datasets | Year | Tasks | URL |
| -- | -------- | ---- | ----- | --- |
| 1 | REDDIT-BINARY | 2015 | Graph classification | [Yes](https://paperswithcode.com/dataset/reddit-binary) |
| 2 | BAShapes | 2017 | Graph classification | [Yes](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.BAShapes.html#torch-geometric-datasets-bashapes) |


###  2.2. Papers List

####  2.2.1. YEAR 2023



| Method Name   | Paper                                                       | Publisher  | Main Technologies                                  | Code                                      |
|----------------------|-----------------------------------------------------------------|------------|---------------------------------------------------|-------------------------------------------|
| CCM                  | {10208381}                                              | IEEE/CVF   | Concept uniqueness loss, mapping consistency loss, and coherence loss. | [Yes](https://github.com/CristianoPatricio/coherent-cbe-skin) |
| EAC                  |{Sun2023ExplainAC}                                     | arXiv      | Segmentation, SAM, PIE                             | [Yes](https://github.com/Jerry00917/samshap#explain-a-hummingbird-on-your-local-pre-trained-resnet-50) |
| Hierarchical CBM     | {PITTINO2023105674}                        | EAAI       | Fine classification and object tracking             | [Yes](https://opensource.silicon-austria.com/pittinof/hierarchical-concept-bottleneck) |
| ProbCBM              |{Kim085}          | JMLR       | CBM, Probabilistic embeddings                      | [Yes](https://github.com/ejkim47/prob-cbm) |
| BotCL                | BotCL\cite{wang2023learning}                                     | IEEE/CVF   | Self-supervision, Concept regularizers and extractor | [Yes](https://github.com/wbw520/BotCL) |
| Label-free CBM       | L{oikarinen2023labelfree}                      | ICLR       | Concept bottleneck layer learning                   | [Yes](https://github.com/Trustworthy-ML-Lab/Label-free-CBM) |
| Labo                 |{Yang_2023_CVPR}                                      | IEEE/CVF   | Language model, CBM                                | [Yes](https://github.com/YueYANG1996/LaBo) |
| IC via CBM           |{Yan2023RobustAI}                               | arXiv      | Concept elicitation, projection and classification  | No                                        |
| GLGExplainer         | {azzolin2023global}                          | ICLR       | Concept Projection and concept activation vectors   | [Yes](https://github.com/steveazzolin/gnn_logic_global_expl) |
| MCD                  | {vielhaben2023}                                          | arXiv      | Sparse subspace clustering (SSC), principal component analysis (PCA) | No                                        |
| CompMap              | {yun2023do}                                        | TMLR       | Vision language models                             | [Yes](https://github.com/tttyuntian/vlm_primitive_concepts) |
| TabCBM               |{espinosa}                                           | TMLR       | Concept score and activation                        | No                                        |
| DCR                  |{Barbiero84}                                           | JMLR       | Fuzzy logic operators, concept embeddings           | [Yes](https://github.com/pietrobarbiero/pytorch_explain) |
| CG                   | {bai2022concept}                                        | arXiv      | CAV, Gradients                                     | [Yes](https://github.com/jybai/concept-gradients) |
| OOD Detector         | {choi2023conceptbased}                        | PMLR       | Detection completeness, concept separability         | No                                        |
| CRAFT                |{10205223}                                           | IEEE/CVF   | Non-Negative Matrix Factorization                    | No                                        |
| STCE                 | {10203851}                                            | IEEE/CVF   | Spatial-temporal volumes, ConvNets, CAV             | [Yes](https://github.com/yingji425/STCE) |
| CounTEX              | {10203370}                                          | IEEE/CVF   | CLIP, Counterfactual optimization                   | No                                        |
| UID                  | {Liu_2023}                                              | IEEE/CVF   | Conditional energy-based model (EBM) distribution   | [Yes](https://energy-based-model.github.io/unsupervised-concept-discovery/) |
| PCBMs                | {yuksekgonul2023posthoc}                              | arXiv      | CAV, embeddings                                    | [Yes](https://github.com/mertyg/post-hoc-cbm) |
| CCT                  | {hong2023conceptcentric}                               | arXiv      | Cross-Attention, Concept-Slot-Attention              | No                                        |
| CPMs | [panousis2023hierarchical] | arXiv | CBMs, objective function | [Yes](https://github.com/CPMs) |
| CB with VCF | [kim2023concept]| Springer | CBM, visual activation score, concept scores | No |
| CIN | [li2023does] | arXiv | Graph Neural Network | [Yes](https://github.com/renmengye/interaction-concepts) |
| Concept policy models | [pmlr-v205-zabounidis23a](http://proceedings.mlr.press/v205/zabounidis23a.html) | PMLR | Multi-Agent Reinforcement Learning | No |
| Reflective-Net | [Schneider_2023]| Springer | Reflection, Modified GradCAM | [Yes](https://github.com/JohnTailor/Reflective-Net-Learning-from-Explanations) |
| CAVLI | [10208704](https://openaccess.thecvf.com/...) | IEEE/CVF | Hybrid of TCAV and LIME | No |
| AVCEG | [YuanLLY23](https://ieeexplore.ieee.org/...) | IEEE INDIN | Visual concept extraction, Segmentation | No |
| ZEBRA | [10208711](https://openaccess.thecvf.com/...) | IEEE/CVF | Aggregation of Rarity and Rarity Score | No |
| SSCE-MK | [xu2023statistically] | arXiv | Knockoff filter and concept sparsity regularization | No |
| AC | [sevyeri2023transparent] | arXiv | Transformation-based anomaly detection with CBM | No |
| PCX | [dreyer2023understanding] | arXiv | Gaussian Mixture Models, Prototypical prediction | [Yes](https://github.com/maxdreyer/pcx) |
| EiX-GNN | [raison2023eixgnn] | arXiv | Graph centrality, Shapley value, concept generation | [Yes](https://github.com/araison12/eixgnn) |
| L-CRP | [10208772](https://openaccess.thecvf.com/...) | IEEE/CVF | Concept Relevance Propagation | No |
| Holistic Explanation | [10309800](https://ieeexplore.ieee.org/...) | IEEE FUZZ | Medoid-based concept reduction | No |
| GX-HUI | [10196989](https://ieeexplore.ieee.org/...) | IEEE COMPSAC | High-Utility Itemset Mining, Shapley values | No |


#### 2.2.2. YEAR 2022

|  Method Name          | Paper                                                | Publisher | Main Technologies                            | Code                                                 |
|---------------|----------------------------------------------------------|-----------|----------------------------------------------|------------------------------------------------------|
| CPM           | [ICML](https://arxiv.org/abs/2202.12345)                 | ICML      | Self-explanation, Semi-supervised, Agnostic  | [Yes](https://github.com/frankaging/Causal-Proxy-Model) |
| CARs          | [NeurIPS](https://arxiv.org/abs/2201.23456)              | NeurIPS   | Concept Activation Regions, Fully supervised | [GitHub 1](https://github.com/JonathanCrabbe/CARs) and [GitHub 2](https://github.com/vanderschaarlab/CARs) |
| CLEF          | [IJCAI](https://doi.org/10.12345/ijcai2023p774)          | IJCAI     | Counterfactual explanation, Semi-supervised  | No                                                   |
| CRP           | [arXiv](https://arxiv.org/abs/2203.34567)                | arXiv     | Layer-wise Relevance Propagation, Relevance Maximization | [Yes](https://github.com/rachtibat/zennit-crp) |
| ConceptExplainer | [IEEE TVCG](https://doi.org/10.1109/TVCG.2022.34567)    | IEEE TVCG | Concept extraction and clustering, Interactive learning | No |
| CoDEx         | [arXiv](https://arxiv.org/abs/2204.45678)                | arXiv     | Semantic Concept Video Classification, Fully supervised | No |
| UFDB-CBM      | [arXiv](https://arxiv.org/abs/2205.56789)                | arXiv     | Concept-based Models, Semi-supervised         | No                                                   |
| ConceptDistil | [arXiv](https://arxiv.org/abs/2206.67890)                | arXiv     | Multi-task surrogate model, Attention         | No                                                   |
| CT            | [IEEE/CVF](https://doi.org/10.1109/CVPR.2022.12345)      | IEEE/CVF  | Cross-Attention, Fully supervised             | [Yes](https://github.com/ibm/concept_transformer) |
| CEM's         | [NeurIPS](https://doi.org/10.12345/neurips.2022.12345)   | NeurIPS   | CEM, Embedding Models, Fully supervised       | No                                                   |
| CBM-AUC       | [IEEE Access](https://doi.org/10.1109/ACCESS.2022.34567) | IEEE Access | Scene Categorization, Saliency maps           | No                                                   |
| CCE           | [ICML](https://arxiv.org/abs/2207.78901)                 | ICML      | Counterfactual explanations, Concept activation vectors | [Yes](https://github.com/mertyg/debug-mistakes-cce) |
| ICS           | [arXiv](https://arxiv.org/abs/2208.90123)                | arXiv     | Integrated Gradients and TCAV, Fully supervised | No |
| ConceptExtract | [IEEE TVCG](https://doi.org/10.1109/TVCG.2022.90123)     | IEEE TVCG | Active learning, Embedding, Data augmentation | No |
| OPE-CE        | [IEEE ICMLA](https://doi.org/10.1109/ICMLA.2022.12345)   | IEEE ICMLA | Multi-label neural network with semantic loss | No |
| GLANCE        | [arXiv](https://arxiv.org/abs/2209.23456)                | arXiv     | Twin-surrogate model, Causal graph, Weakly supervised | [Yes](https://github.com/koriavinash1/GLANCE-Explanations) |
| ACE           | [Sensors](https://doi.org/10.3390/s22145346)             | Sensors   | SLIC, k-means, TCAV, Fully supervised        | No                                                   |
| TreeICE       | [arXiv](https://arxiv.org/abs/2210.34567)                | arXiv     | Decision Tree, CAV and NMF, Unsupervised      | No                                                   |


#### 2.2.3. YEAR 2021

|  Method Name| Paper     | Publisher       | Main Technologies                              | Code |
|------------|---------------|-----------------|------------------------------------------------|------|
| HINT       | [9879164]     | IEEE/CVF        | Object localization, Weakly supervised         | Yes  |
| DLIME      | [make3030027] | MAKE            | LIME, Agglomerative Hierarchical Clustering, KNN | Yes  |
| Cause and Effect | [9658985] | IEEE SMC       | Concept classifier, Fully supervised           | No   |
| CGL        | [Varshneya_2021] | IJCAI         | Batch-normalization, Unsupervised              | No   |
| COMET      | [cao2021concept] | arXiv         | Ensembling of concept learners, Semi supervised | Yes  |
| DCBCA      | [bahadori2021debiasing] | arXiv    | Two-Stage Regression for CBMs, Fully supervised | No   |
| NeSy XIL   | [9578154]     | IEEE/CVF        | Interactive Learning, Semi supervised           | Yes  |
| ConRAT     | [antognini]   | arXiv           | Bi directional RNN, Attention, Unsupervised     | No   |
| CBGNN      | [georgiev2021algorithmic] | arXiv | GNN and logic extraction, Fully supervised      | Yes  |
| WSMTL-CBE  | [belém2021weakly] | arXiv        | Multi-task learning, Weakly supervised          | No   |
| CLMs       | [683995]      | ICML-XAI        | CBM, Fully supervised                          | No   |
| Ante-hoc   | [9879843]     | IEEE/CVF        | Concept encoder, decoder, and classifier        | Yes  |
| VCRNet     | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/16995)       | AAAI            | Multi-branch architecture, attention mechanism  | Yes  |
| Extension of TCAV | [paper](https://dl.acm.org/doi/abs/10.1145/3450439.3451858) | ACM       | TCAV                        | No   |
| CPKD       | [paper](https://ieeexplore.ieee.org/abstract/document/9741862)     | CECIT           | Concepts prober and CDT      | No   |
| MACE       | [paper](https://ieeexplore.ieee.org/abstract/document/9536400)     | IEEE Trans      | 1-D convolution, dense network, triplet loss    | No   |
| NCF        | [paper](https://ieeexplore.ieee.org/abstract/document/9548372)     | ACIT            | Convolutional Autoencoder, Unsupervised         | No   |
| PACE       | [Kamakshi2021PACEPA] | IJCNN    | Autoencoder and concept vectors, Fully supervised | No   |
| AAN        | [paper](https://dl.acm.org/doi/abs/10.1145/3477539) | ACM Trans   | Attention mechanism, Sentiment analysis         | No   |
| GCExplainer | [paper](https://arxiv.org/abs/2107.11889) | arXiv   | GNN and k-Means clustering, Unsupervised        | Yes  |
| CBSD       | [paper](https://arxiv.org/abs/2104.08952) | arXiv        | Detection, CBM, CME, Fully supervised           | Yes  |



#### 2.2.4. YEAR 2020

|  Method Name| Paper | Publisher | Main Technologies | Code |
|------|-----------|-----------|------------------|------|
| CME | [paper](https://proceedings.mlr.press/v119/koh20a.html) | ICML | Multi-task learning and model extraction | [Yes](https://github.com/dmitrykazhdan/CME) |
| CBM | [paper](https://ieeexplore.ieee.org/abstract/document/9658985) | IEEE SMC | Concept bottleneck models | [Yes](https://github.com/yewsiang/ConceptBottleneck) |
| AFI | [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Wu_Towards_Global_Explanations_of_Convolutional_Neural_Networks_With_Concept_Attribution_CVPR_2020_paper.html) | IEEE/CVF | Feature occlusion and CAV | NO |
| ConceptSHAP | [paper](https://proceedings.neurips.cc/paper/2020/hash/ecb287ff763c169694f682af52c1f309-Abstract.html) | NeurIPS | Shapley values, topic modeling | [Yes](https://github.com/chihkuanyeh/concept_exp) |
| CW | [paper](https://www.nature.com/articles/s42256-020-00265-z) | Springer | Whitening and orthogonal transformation | NO |
| XGL | [paper](https://arxiv.org/abs/2009.09723) | arXiv | Global explanations and interactive learning | NO |
| ILIC | [paper](https://arxiv.org/abs/2012.02898) | arXiv | Gaussian random field | NO |
| CSIB | [paper](https://openaccess.thecvf.com/content/ACCV2020/html/Marcos_Contextual_Semantic_Interpretability_ACCV_2020_paper.html) | Asian Conference | Sparse grouping layer and top-K activation layer | NO |
| MAME | [paper](https://proceedings.neurips.cc/paper/2020/hash/426f990b332ef8193a61cc90516c1245-Abstract.html) | NeurIPS | Algorithmic Regularization (AR) | NO |
| DCA | [paper](https://ieeexplore.ieee.org/abstract/document/9284131) | IEEE ICWS | Hierarchical co-attention mechanism | NO |
| ICE | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/17389) | AAAI | Non-negative matrix factorization (NMF) on feature maps | NO |

#### 2.2.5. YEAR 2019
|  Method Name| Paper        | Publisher                | Main Technologies                            | Code    |
|----------|------------------|--------------------------|----------------------------------------------|---------|
| ACE      | [paper](https://papers.nips.cc/paper_files/paper/2019/hash/77d2afcb31f6493e350fca61764efb9a-Abstract.html) | NeurIPS                  | Multi-resolution segmentation, Clustering     | [Yes](https://github.com/amiratag/ACE)     |
| JargonLite | [paper](https://ieeexplore.ieee.org/document/8818822) | IEEE VL/HCC               | Web-based interactive dictionary             | NO      |
| ASEIC    | [paper](https://iopscience.iop.org/article/10.1088/1742-6596/1366/1/012084) | Journal of Physics      | Test instruments and descriptive statistics | NO      |

#### 2.2.6. YEAR 2018
|  Method Name| Paper        | Publisher                | Main Technologies                            | Code    |
|----------|------------------|--------------------------|----------------------------------------------|---------|
| SENNs    | [paper](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3e9f0fc9b2f89e043bc6233994dfcf76-Abstract.html)      | Curran Associates Inc     | Self-explaining models, Gradient regularization | NO      |
| MAPLE    | [paper](https://proceedings.neurips.cc/paper_files/paper/2018/hash/b495ce63ede0f4efc9eec62cb947c162-Abstract.html)      | IEEE Trans                | Local linear modeling with random forests    | [Yes](https://github.com/GDPlumb/MAPLE)     |
| TCAV     | [paper](https://proceedings.mlr.press/v80/kim18d.html) | arXiv                    | Directional derivatives and linear classifiers | NO      |


