# LDA-GCL

## This is our PyTorch implementation code for our paper:
> Adversarial Learning Data Augmentation for Graph Contrastive Learning in Recommendation (DASFAA2023)
> 
> [arXiv](https://arxiv.org/abs/2302.02317)


## Methods


![](https://huangjunjie-cs.github.io/static/uploads/covers/LDA_GCL.png)
LDA-GCL includes learning data augmentation and graph contrastive learning.



## Dependency

In order to run this code, you need to install following dependencies:

```
pip install -r requirements.txt
```

## Run Experiments

### Data split
Before running the code on your own dataset, it's recommended to take some steps to ensure consistency and validity of the results. 
First, it's suggested to perform data splitting beforehand instead of relying on the random seed in the [code](https://github.com/RUCAIBox/NCL). 

Also, note that the approach described in the paper requires training a model to obtain potential edges, and then combining them with the original edge set to create a candidate edge set. 
To this end, you can use the ```LightGCN```  to train a model and save its parameters. 
In case you don't want to train the model from scratch, a pre-trained model is provided as an example in the code. 
You can download it from the link provided in the instructions and save it in the ```saved``` directory.

### Run 
You can run following commands:

```
python run_lda_gcl.py -m LDAGCL --dataset gowalla-merged --adv_reg=0.2 --learn_flag=True --benchmark-filename='["train-1", "valid-1", "test-1"]'
```

The results are listed as follows:

```
...

OrderedDict([('recall@10', 0.1342), ('recall@20', 0.1966), ('recall@50', 0.3089), ('ndcg@10', 0.0962), ('ndcg@20', 0.1139), ('ndcg@50', 0.1413)])
# this is the performance of original LightGCN
...

14 Mar 01:09    INFO  best valid : OrderedDict([('recall@10', 0.1483), ('recall@20', 0.2135), ('recall@50', 0.3309), ('ndcg@10', 0.1045), ('ndcg@20', 0.1235), ('ndcg@50', 0.1523)])
14 Mar 01:09    INFO  test result: OrderedDict([('recall@10', 0.1518), ('recall@20', 0.2137), ('recall@50', 0.3283), ('ndcg@10', 0.1092), ('ndcg@20', 0.1268), ('ndcg@50', 0.1548)])
# this is the final performance of LDA-GCL
```


## Citation

Please cite our paper if you find this code can be helpful in your work

```
@article{huang2023adversarial,
  title={Adversarial Learning Data Augmentation for Graph Contrastive Learning in Recommendation},
  author={Huang, Junjie and Cao, Qi and Xie, Ruobing and Zhang, Shaoliang and Xia, Feng and Shen, Huawei and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2302.02317},
  year={2023}
}
```
