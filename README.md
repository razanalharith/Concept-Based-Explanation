# Concept-Based-Explanation


Concept-Based-Explanation methods.

<p align="center">
    <img src="polyp.png"/> <br />
</p>

# <p align=center>`Concept-Based-Explanation`


< **Last updated: 29/12/2023** >


##  1. Content


- [List for Concept-Based-Explanation methods]
	- [1. Content](#1-content)
	- [2. Paper List](#2-paper-list)
		- [2.1. Datasets 

		- [2.2. Concept-Based-Explanation methods 
			- [2.2.1. YEAR 2023](#221-year-2023)
			- [2.2.2. YEAR 2022](#222-year-2022)
			- [2.2.3. YEAR 2021](#223-year-2021)
			- [2.2.4. Before YEAR 2020](#224-before-year-2020)
     
		
    

##  2. Paper List

###  2.1. Datasets 

|                   | Datasets                             | Year  | Size       | Tasks                   | URL                                                                                      | Reference                                                                                                          |
|-------------------|--------------------------------------|-------|------------|-------------------------|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **Image**         |                                      |       |            |                         |                                                                                          |                                                                                                                    |
|                   | ImageNet                             | 2009  | 14,197,122 | Classification + Detection | [Yes](http://www.image-net.org/)                                                          | [1, 2, 3, 4, 5, 6]                                                                                                  |
|                   | Derm7pt                              | 2019  | 2K         | Detection               | [Yes](https://derm.cs.sfu.ca/Welcome.html)                                               | [7]                                                                                                                |
|                   | Microsoft COCO                       | 2014  | 328K       | Detection               | [Yes](https://cocodataset.org/)                                                           | [5, 8, 9]                                                                                                          |
|                   | Caltech-UCSD Birds-200-2011 (CUB)     | 2011  | 11,788     | Classification          | [Yes](https://www.vision.caltech.edu/datasets/cub_200_2011/)                            | [10, 11, 2, 12, 13, 14]                                                                                             |
|                   | Animal with Attributes (AwA2)         | 2018  | 37,322     | Classification          | [Yes](https://paperswithcode.com/dataset/awa2-1)                                         | [4, 15, 16, 17]                                                                                                    |
|                   | CIFAR-10 and CIFAR-100                | 2009  | 80M        | Classification          | [Yes](https://www.cs.toronto.edu/~kriz/cifar.html)                                       | [4]                                                                                                                |
|                   | Places365                            | 2015  | 1.8 M      | Scene Recognition        | [Yes](https://paperswithcode.com/dataset/places365)                                      | [18, 3, 19, 16]                                                                                                    |
|                   | Skin Cancer MNIST: HAM10000           | 2018  | 10K        | Detection               | [Yes](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)                 | [20, 7, 21]                                                                                                        |
|                   | NIH Chest X-ray                       | 2019  | 112,120    | Detection               | [Yes](https://datasets.activeloop.ai/docs/ml/datasets/nih-chest-x-ray-dataset/)          | [22]                                                                                                               |
|                   | MIT-States                           | 2015  | 53K        | Zero Shot Learning       | [Yes](http://web.mit.edu/phillipi/Public/states_and_transformations/index.html)           | [23]                                                                                                               |
|                   | MNIST                                | 2010  | 70K        | Classification          | [Yes](http://yann.lecun.com/exdb/mnist/)                                                 | [24, 25, 26]                                                                                                       |
|                   | Myocardial infarction complications   | 2020  | 1,700      | Classification          | [Yes](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications)       | [27]                                                                                                               |
|                   | ADE20K                               | 2017  | 23K        | Segmentation             | [Yes](https://groups.csail.mit.edu/vision/datasets/ADE20K/)                               | [28]                                                                                                               |
|                   | SIIM-ISIC                            | 2020  | 33,126     | Detection               | [Yes](https://www.kaggle.com/c/siim-isic-melanoma-classification/overview)                 | [29]                                                                                                               |
|                   | CelebA                               | 2018  | 200K       | Detection               | [Yes](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)                                   | [30, 31, 32]                                                                                                       |
|                   | SUN attribute                         | 2010  | 130,519    | Scene Categorization     | [Yes](https://vision.princeton.edu/projects/2010/SUN/)                                   | [33, 34]                                                                                                           |
|                   | IP102                                | 2019  | 75K        | Classification          | [Yes](https://github.com/xpwu95/IP102)                                                    | [35]                                                                                                               |
|                   | MONK's                               | 1992  | I apologize for the incomplete response. Here's the complete modified table in full GitHub Markdown format:

```markdown
|                   | Datasets                             | Year  | Size       | Tasks                   | URL                                                                                      | Reference                                                                                                          |
|-------------------|--------------------------------------|-------|------------|-------------------------|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **Image**         |                                      |       |            |                         |                                                                                          |                                                                                                                    |
|                   | ImageNet                             | 2009  | 14,197,122 | Classification + Detection | [Yes](http://www.image-net.org/)                                                          | [1, 2, 3, 4, 5, 6]                                                                                                  |
|                   | Derm7pt                              | 2019  | 2K         | Detection               | [Yes](https://derm.cs.sfu.ca/Welcome.html)                                               | [7]                                                                                                                |
|                   | Microsoft COCO                       | 2014  | 328K       | Detection               | [Yes](https://cocodataset.org/)                                                           | [5, 8, 9]                                                                                                          |
|                   | Caltech-UCSD Birds-200-2011 (CUB)     | 2011  | 11,788     | Classification          | [Yes](https://www.vision.caltech.edu/datasets/cub_200_2011/)                            | [10, 11, 2, 12, 13, 14]                                                                                             |
|                   | Animal with Attributes (AwA2)         | 2018  | 37,322     | Classification          | [Yes](https://paperswithcode.com/dataset/awa2-1)                                         | [4, 15, 16, 17]                                                                                                    |
|                   | CIFAR-10 and CIFAR-100                | 2009  | 80M        | Classification          | [Yes](https://www.cs.toronto.edu/~kriz/cifar.html)                                       | [4]                                                                                                                |
|                   | Places365                            | 2015  | 1.8 M      | Scene Recognition        | [Yes](https://paperswithcode.com/dataset/places365)                                      | [18, 3, 19, 16]                                                                                                    |
|                   | Skin Cancer MNIST: HAM10000           | 2018  | 10K        | Detection               | [Yes](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)                 | [20, 7, 21]                                                                                                        |
|                   | NIH Chest X-ray                       | 2019  | 112,120    | Detection               | [Yes](https://datasets.activeloop.ai/docs/ml/datasets/nih-chest-x-ray-dataset/)          | [22]                                                                                                               |
|                   | MIT-States                           | 2015  | 53K        | Zero Shot Learning       | [Yes](http://web.mit.edu/phillipi/Public/states_and_transformations/index.html)           | [23]                                                                                                               |
|                   | MNIST                                | 2010  | 70K        | Classification          | [Yes](http://yann.lecun.com/exdb/mnist/)                                                 | [24, 25, 26]                                                                                                       |
|                   | Myocardial infarction complications   | 2020  | 1,700      | Classification          | [Yes](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications)       | [27]                                                                                                               |
|                   | ADE20K                               | 2017  | 23K        | Segmentation             | [Yes](https://groups.csail.mit.edu/vision/datasets/ADE20K/)                               | [28]                                                                                                               |
|                   | SIIM-ISIC                            | 2020  | 33,126     | Detection               | [Yes](https://www.kaggle.com/c/siim-isic-melanoma-classification/overview)                 | [29]                                                                                                               |
|                   | CelebA                               | 2018  | 200K       | Detection               | [Yes](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)                                   | [30, 31, 32]                                                                                                       |
|                   | SUN attribute                         | 2010  | 130,519    | Scene Categorization     | [Yes](https://vision.princeton.edu/projects/2010/SUN/)                                   | [33, 34]                                                                                                           |
|                   | IP102                                | 2019  | 75K        | Classification          | [Yes](https://github.com/xpwu95/IP102)                                                    | [35]                                                                                                               |
|                   | MONK's                               |



###  2.2. Papers


####  2.2.1. YEAR 2023
