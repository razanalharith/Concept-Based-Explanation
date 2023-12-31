
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
| CCM                  | [paper](https://openaccess.thecvf.com/content/CVPR2023W/SAIAD/html/Patricio_Coherent_Concept-Based_Explanations_in_Medical_Image_and_Its_Application_to_CVPRW_2023_paper.html)                                             | IEEE/CVF   | Concept uniqueness loss, mapping consistency loss, and coherence loss. | [Yes](https://github.com/CristianoPatricio/coherent-cbe-skin) |
| EAC                  |[paper](https://arxiv.org/abs/2305.10289)                                     | arXiv      | Segmentation, SAM, PIE                             | [Yes](https://github.com/Jerry00917/samshap#explain-a-hummingbird-on-your-local-pre-trained-resnet-50) |
| Hierarchical CBM     | [paper](https://www.sciencedirect.com/science/article/pii/S0952197622006649)                        | EAAI       | Fine classification and object tracking             | [Yes](https://opensource.silicon-austria.com/pittinof/hierarchical-concept-bottleneck) |
| ProbCBM              |[paper](https://arxiv.org/abs/2306.01574)          | JMLR       | CBM, Probabilistic embeddings                      | [Yes](https://github.com/ejkim47/prob-cbm) |
| BotCL                | [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Learning_Bottleneck_Concepts_in_Image_Classification_CVPR_2023_paper.html)                                     | IEEE/CVF   | Self-supervision, Concept regularizers and extractor | [Yes](https://github.com/wbw520/BotCL) |
| Label-free CBM       | [paper](https://arxiv.org/abs/2304.06129)                      | ICLR       | Concept bottleneck layer learning                   | [Yes](https://github.com/Trustworthy-ML-Lab/Label-free-CBM) |
| Labo                 |[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Language_in_a_Bottle_Language_Model_Guided_Concept_Bottlenecks_for_CVPR_2023_paper.html)                                  | IEEE/CVF   | Language model, CBM                                | [Yes](https://github.com/YueYANG1996/LaBo) |
| IC via CBM           |[paper](https://arxiv.org/abs/2310.03182)                   | arXiv      | Concept elicitation, projection and classification  | No                                        |
| GLGExplainer         | [paper](https://arxiv.org/abs/2210.07147)                          | ICLR       | Concept Projection and concept activation vectors   | [Yes](https://github.com/steveazzolin/gnn_logic_global_expl) |
| MCD                  |[paper](https://arxiv.org/abs/2301.11911)                                  | arXiv      | Sparse subspace clustering (SSC), principal component analysis (PCA) | No                                        |
| CompMap              | [paper](https://arxiv.org/abs/2203.17271)                                        | TMLR       | Vision language models                             | [Yes](https://github.com/tttyuntian/vlm_primitive_concepts) |
| TabCBM               |[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/867c06823281e506e8059f5c13a57f75-Abstract-Conference.html)                                           | TMLR       | Concept score and activation                        | No                                        |
| DCR                  |[paper](https://arxiv.org/abs/2304.14068)                                         | JMLR       | Fuzzy logic operators, concept embeddings           | [Yes](https://github.com/pietrobarbiero/pytorch_explain) |
| CG                   | [paper](https://arxiv.org/abs/2208.14966)                                     | arXiv      | CAV, Gradients                                     | [Yes](https://github.com/jybai/concept-gradients) |
| OOD Detector         |[paper](https://proceedings.mlr.press/v202/choi23e.html)                        | PMLR       | Detection completeness, concept separability         | No                                        |
| CRAFT                |[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Fel_CRAFT_Concept_Recursive_Activation_FacTorization_for_Explainability_CVPR_2023_paper.html)                                           | IEEE/CVF   | Non-Negative Matrix Factorization                    | No                                        |
| STCE                 |[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Ji_Spatial-Temporal_Concept_Based_Explanation_of_3D_ConvNets_CVPR_2023_paper.html)                                            | IEEE/CVF   | Spatial-temporal volumes, ConvNets, CAV             | [Yes](https://github.com/yingji425/STCE) |
| CounTEX              | [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_Grounding_Counterfactual_Explanation_of_Image_Classifiers_to_Textual_Concept_Space_CVPR_2023_paper.html)                                        | IEEE/CVF   | CLIP, Counterfactual optimization                   | No                                        |
| UID                  | [paper](https://arxiv.org/abs/2306.05357)                                              | IEEE/CVF   | Conditional energy-based model (EBM) distribution   | [Yes](https://energy-based-model.github.io/unsupervised-concept-discovery/) |
| PCBMs                |[paper](https://arxiv.org/abs/2205.15480)               | arXiv      | CAV, embeddings                                    | [Yes](https://github.com/mertyg/post-hoc-cbm) |
| CCT                  |[paper](https://openaccess.thecvf.com/content/WACV2024/html/Hong_Concept-Centric_Transformers_Enhancing_Model_Interpretability_Through_Object-Centric_Concept_Learning_Within_WACV_2024_paper.html)                               | arXiv      | Cross-Attention, Concept-Slot-Attention              | No                                        |
| CPMs |[paper](https://arxiv.org/abs/2310.02116) | arXiv | CBMs, objective function | [Yes](https://github.com/CPMs) |
| CB with VCF |[paper](https://link.springer.com/chapter/10.1007/978-3-031-47401-9_22)| Springer | CBM, visual activation score, concept scores | No |
| CIN |[paper](https://arxiv.org/abs/2302.13080) | arXiv | Graph Neural Network | [Yes](https://github.com/renmengye/interaction-concepts) |
| Concept policy models |[paper](https://proceedings.mlr.press/v205/zabounidis23a.html) | PMLR | Multi-Agent Reinforcement Learning | No |
| Reflective-Net | [paper](https://link.springer.com/article/10.1007/s10618-023-00920-0)| Springer | Reflection, Modified GradCAM | [Yes](https://github.com/JohnTailor/Reflective-Net-Learning-from-Explanations) |
| CAVLI | [paper](https://openaccess.thecvf.com/content/CVPR2023W/XAI4CV/html/Shukla_CAVLI_-_Using_Image_Associations_To_Produce_Local_Concept-Based_Explanations_CVPRW_2023_paper.html) | IEEE/CVF | Hybrid of TCAV and LIME | No |
| AVCEG |[paper](https://ieeexplore.ieee.org/abstract/document/10217975) | IEEE INDIN | Visual concept extraction, Segmentation | No |
| ZEBRA |[paper](https://openaccess.thecvf.com/content/CVPR2023W/XAI4CV/html/Madeira_ZEBRA_Explaining_Rare_Cases_Through_Outlying_Interpretable_Concepts_CVPRW_2023_paper.html) | IEEE/CVF | Aggregation of Rarity and Rarity Score | No |
| SSCE-MK |[paper](https://openaccess.thecvf.com/content/CVPR2023W/XAI4CV/html/Madeira_ZEBRA_Explaining_Rare_Cases_Through_Outlying_Interpretable_Concepts_CVPRW_2023_paper.html) | arXiv | Knockoff filter and concept sparsity regularization | No |
| AC |[paper](https://arxiv.org/abs/2310.10702) | arXiv | Transformation-based anomaly detection with CBM | No |
| PCX |[paper](https://arxiv.org/abs/2311.16681) | arXiv | Gaussian Mixture Models, Prototypical prediction | [Yes](https://github.com/maxdreyer/pcx) |
| EiX-GNN |[paper](https://arxiv.org/abs/2206.03491) | arXiv | Graph centrality, Shapley value, concept generation | [Yes](https://github.com/araison12/eixgnn) |
| L-CRP |[paper](https://openaccess.thecvf.com/content/CVPR2023W/SAIAD/html/Dreyer_Revealing_Hidden_Context_Bias_in_Segmentation_and_Object_Detection_Through_CVPRW_2023_paper.html)| IEEE/CVF | Concept Relevance Propagation | No |
| Holistic Explanation | [paper](https://ieeexplore.ieee.org/abstract/document/10309800) | IEEE FUZZ | Medoid-based concept reduction | No |
| GX-HUI |[paper](https://ieeexplore.ieee.org/abstract/document/10196989) | IEEE COMPSAC | High-Utility Itemset Mining, Shapley values | No |


#### 2.2.2. YEAR 2022

|  Method Name          | Paper                                                | Publisher | Main Technologies                            | Code                                                 |
|---------------|----------------------------------------------------------|-----------|----------------------------------------------|------------------------------------------------------|
| CPM           | [paper](https://proceedings.mlr.press/v202/wu23b.html)                 | ICML      | Self-explanation, Semi-supervised, Agnostic  | [Yes](https://github.com/frankaging/Causal-Proxy-Model) |
| CARs          | [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/11a7f429d75f9f8c6e9c630aeb6524b5-Abstract-Conference.html)             | NeurIPS   | Concept Activation Regions, Fully supervised | [GitHub 1](https://github.com/JonathanCrabbe/CARs) and [GitHub 2](https://github.com/vanderschaarlab/CARs) |
| CLEF          |[paper](https://www.jair.org/index.php/jair/article/view/14019)        | IJCAI     | Counterfactual explanation, Semi-supervised  | No                                                   |
| CRP           | [paper](https://arxiv.org/abs/2206.03208)           | arXiv     | Layer-wise Relevance Propagation, Relevance Maximization | [Yes](https://github.com/rachtibat/zennit-crp) |
| ConceptExplainer |[paper](https://ieeexplore.ieee.org/abstract/document/9903285)   | IEEE TVCG | Concept extraction and clustering, Interactive learning | No |
| CoDEx         | [paper](https://arxiv.org/abs/2206.10129)              | arXiv     | Semantic Concept Video Classification, Fully supervised | No |
| UFDB-CBM      | [paper](https://arxiv.org/abs/2109.11160)             | arXiv     | Concept-based Models, Semi-supervised         | No                                                   |
| ConceptDistil | [paper](https://arxiv.org/abs/2205.03601)              | arXiv     | Multi-task surrogate model, Attention         | No                                                   |
| CT            | [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html)     | IEEE/CVF  | Cross-Attention, Fully supervised             | [Yes](https://github.com/ibm/concept_transformer) |
| CEM's         | [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/867c06823281e506e8059f5c13a57f75-Abstract-Conference.html)   | NeurIPS   | CEM, Embedding Models, Fully supervised       | No                                                   |
| CBM-AUC       | [paper](https://ieeexplore.ieee.org/abstract/document/9758745)| IEEE Access | Scene Categorization, Saliency maps           | No                                                   |
| CCE           | [paper](https://proceedings.mlr.press/v162/abid22a.html)              | ICML      | Counterfactual explanations, Concept activation vectors | [Yes](https://github.com/mertyg/debug-mistakes-cce) |
| ICS           | [paper](https://arxiv.org/abs/2106.08641)                | arXiv     | Integrated Gradients and TCAV, Fully supervised | No |
| ConceptExtract | [paper](https://ieeexplore.ieee.org/abstract/document/9552218)    | IEEE TVCG | Active learning, Embedding, Data augmentation | No |
| OPE-CE        | [paper](https://ieeexplore.ieee.org/abstract/document/10069035)   | IEEE ICMLA | Multi-label neural network with semantic loss | No |
| GLANCE        | [paper](https://arxiv.org/abs/2207.01917)                | arXiv     | Twin-surrogate model, Causal graph, Weakly supervised | [Yes](https://github.com/koriavinash1/GLANCE-Explanations) |
| ACE           | [paper](https://www.mdpi.com/1424-8220/22/14/5346)            | Sensors   | SLIC, k-means, TCAV, Fully supervised        | No                                                   |
| TreeICE       | [paper](https://arxiv.org/abs/2211.10807)        | arXiv     | Decision Tree, CAV and NMF, Unsupervised      | No                                                   |


#### 2.2.3. YEAR 2021

|  Method Name| Paper     | Publisher       | Main Technologies                              | Code |
|------------|---------------|-----------------|------------------------------------------------|------|
| HINT       | [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_HINT_Hierarchical_Neuron_Concept_Explainer_CVPR_2022_paper.html)    | IEEE/CVF        | Object localization, Weakly supervised         | Yes  |
| DLIME      | [paper](https://www.mdpi.com/2504-4990/3/3/27) | MAKE            | LIME, Agglomerative Hierarchical Clustering, KNN | Yes  |
| Cause and Effect | [paper](https://ieeexplore.ieee.org/abstract/document/9658985) | IEEE SMC       | Concept classifier, Fully supervised           | No   |
| CGL        | [paper](https://arxiv.org/abs/2109.10078) | IJCAI         | Batch-normalization, Unsupervised              | No   |
| COMET      | [paper](https://arxiv.org/abs/2007.07375) | arXiv         | Ensembling of concept learners, Semi supervised | Yes  |
| DCBCA      | [paper](https://arxiv.org/abs/2007.11500) | arXiv    | Two-Stage Regression for CBMs, Fully supervised | No   |
| NeSy XIL   | [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Stammer_Right_for_the_Right_Concept_Revising_Neuro-Symbolic_Concepts_by_Interacting_CVPR_2021_paper.html)     | IEEE/CVF        | Interactive Learning, Semi supervised           | Yes  |
| ConRAT     | [paper](https://arxiv.org/abs/2105.04837)   | arXiv           | Bi directional RNN, Attention, Unsupervised     | No   |
| CBGNN      | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20623) | arXiv | GNN and logic extraction, Fully supervised      | Yes  |
| WSMTL-CBE  | [paper](https://arxiv.org/abs/2104.12459) | arXiv        | Multi-task learning, Weakly supervised          | No   |
| CLMs       | [paper](https://arxiv.org/abs/2106.13314)      | ICML-XAI        | CBM, Fully supervised                          | No   |
| Ante-hoc   | [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Sarkar_A_Framework_for_Learning_Ante-Hoc_Explainable_Models_via_Concepts_CVPR_2022_paper.html)     | IEEE/CVF        | Concept encoder, decoder, and classifier        | Yes  |
| VCRNet     | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/16995)       | AAAI            | Multi-branch architecture, attention mechanism  | Yes  |
| Extension of TCAV | [paper](https://dl.acm.org/doi/abs/10.1145/3450439.3451858) | ACM       | TCAV                        | No   |
| CPKD       | [paper](https://ieeexplore.ieee.org/abstract/document/9741862)     | CECIT           | Concepts prober and CDT      | No   |
| MACE       | [paper](https://ieeexplore.ieee.org/abstract/document/9536400)     | IEEE Trans      | 1-D convolution, dense network, triplet loss    | No   |
| NCF        | [paper](https://ieeexplore.ieee.org/abstract/document/9548372)     | ACIT            | Convolutional Autoencoder, Unsupervised         | No   |
| PACE       | [paper](https://ieeexplore.ieee.org/abstract/document/9534369) | IJCNN    | Autoencoder and concept vectors, Fully supervised | No   |
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


