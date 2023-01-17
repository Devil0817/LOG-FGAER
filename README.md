READ ME

# LOG-FGAER
a logic-guided fine-grained address recognition method (Log-FGAER), where we formulate the address hierarchy relationship as the logic rule and softly apply it in a probabilistic manner to improve the accuracy of FGAER


# 导航 Table of contents

* [安装](#安装)
* [NER任务](#soft logic模型及其他基线模型)



# 安装 Setup

##### 1. 安装PaddlePaddle install PaddlePaddle 

本项目依赖PaddlePaddle 1.7.0+， 请参考[这里](https://www.paddlepaddle.org.cn/install/quick)安装 PaddlePaddle。

##### 2. 安装ERNIE套件 install ernie


```script
pip install paddle-ernie
```

或者

```shell
git clone https://github.com/PaddlePaddle/ERNIE.git --depth 1
cd ERNIE
pip install -r requirements.txt
pip install -e .
```
`propeller`是辅助模型训练的高级框架，包含NLP常用的前、后处理流程。你可以通过将本repo根目录放入`PYTHONPATH`的方式导入`propeller`:
```shell
export PYTHONPATH=$PWD:$PYTHONPATH
```

##### 3. 数据集 datasets
数据目录整理成以下格式，方便后续使用（通过`--data_dir`参数将数据路径传入训练脚本）；

the `--data_dir` option in the following section assumes a directory tree like this:

```shell
soft_logic/data/dialogue
├── dev
│   └── 1
├── test
│   └── 1
└── train
    └── 1
```
	
构建数据集Dialogue-AER存放在./soft_logic/data/dialogue中，真实下游数据集存放在./soft_logic/data/cucc中

数据示例如下：
![7d5fa839bef0ee46f87efb5e3995aac](https://user-images.githubusercontent.com/44054130/212827608-f55f87b6-d68e-4e3c-a95e-ed9d779ff885.png)


#####  4. 环境 environment
需要配置环境在requirements.txt中 pip install -r requirements.txt进行安装

# NER任务 Run task

##### dialogue_crf_sl.py #soft logic模型, soft_logic目录下 
运行： python3 dialogue_crf_sl.py \
              --from_pretrained ernie-1.0 \
              --data_dir ./data/dialogue \
              --max_steps #set this to EPOCH * NUM_SAMPLES / BATCH_SIZE \
              --save_dir ./save
	   
##### dialogue_bilstm_crf.py #baseline ERNIE-BiLSTM-CRF模型, soft_logic目录下 
##### dialogue_crf.py #baseline ERNIE-CRF模型, soft_logic目录下 
##### dialogue.py #baseline ERNIE模型, soft_logic目录下

运行日志保存在./log中


# 文献引用 Reference

##### ERNIE 1.0
```
@article{sun2019ernie,
  title={Ernie: Enhanced representation through knowledge integration},
  author={Sun, Yu and Wang, Shuohuan and Li, Yukun and Feng, Shikun and Chen, Xuyi and Zhang, Han and Tian, Xin and Zhu, Danxiang and Tian, Hao and Wu, Hua},
  journal={arXiv preprint arXiv:1904.09223},
  year={2019}
}
```
