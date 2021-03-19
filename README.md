# S2VD
# Semi-supervised Video Deraining with Dynamical Rain Generator (CVPR, 2021) [arXiv](https://arxiv.org/abs/2103.07939)

# Requirements and Dependencies
* Ubuntu 16.04, cuda 10.0
* Python 3.6.10, Pytorch 1.6.0
* More detail (See [environment.yml](environment.yml))

# Training pipelines
1. Download the NTURain dataset from [here](https://github.com/hotndy/SPAC-SupplementaryMaterials), and prepare the training data as follows:
    - Labled synthetic data:
        ```python
            python makedata/preparedata_NTU.py  --ntu_path your_downloaded_synthetic_path --train_path your_saved_train_path 
        ```

    - Unlabled real data:

        ```python
            python makedata/preparedata_NTU_semi.py  --ntu_path_semi your_downloaded_real_path --train_path your_saved_train_path
        ```

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Note that you should better put the synthetic and real training data sets into two different training folders.

2. Modify the configured file [options_derain.json](options_derain.json) according to your own training and testing path. 

3. Begin training:

    ```
        python main_NTURain.py
    ```

# Testing pipelines
You need firstly download the testing dataset of [NTURain](https://github.com/hotndy/SPAC-SupplementaryMaterials) and [MSCSC](MSCS://github.com/MinghanLi/MS-CSC-Rain-Streak-Removal) into the folder [testsets](testsets).

+ NTURain synthetic data set:
    ```
        python test_NTURain_synthetic.py
    ```

    This manuscript will re-produce the paper results in Table 1. 


+ NTURain real data set:
    ```
        python test_NTURain_real.py
    ```

+ MSCSC real data set:
    ```
        python test_MSCSC_real.py
    ```


# Citation
```
@incollection{ECCV2020_984,
title = {Semi-supervised video deraining with dynamical rain generator},
author = {Yue, Zongsheng and Xie, Jianwen and Zhao, Qian and Meng, Deyu},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year = {2021}
}
```
