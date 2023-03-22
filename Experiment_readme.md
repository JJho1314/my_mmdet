#  实验记录

## 1. 实验

### 1.1 基于LVIS数据集：

### 1.2 基于MSCOCO数据集：

#### 1.2.1 数据集划分：

参照detpro，将coco划分为base（48类）和novel（17类）其余为不可见类（15类）

#### 1.2.2 实验结果：

| 实验序号        | 描述                                            | Novel  AP         | mAP  |
| --------------- | ----------------------------------------------- | ----------------- | ---- |
| f_vlm_coco_v1.0 | 修改$\alpha$=0.2，$\beta$=0.45；修改softmax位置 | 12                | 34.5 |
| f_vlm_coco_v2.0 | 修改训练策略，nms_thres，                       | 1.4（零样本检测） | /    |
| f_vlm_coco_v3.0 | 修改self.ignore_cats                            | 27.2              | 39.5 |



## 2. 对比实验

### 2.1 基于MSCOCO数据集

| Backbone     | Head          | Trainable Backbone | Training source | bbox_mAP | Novel AP |
| ------------ | ------------- | ------------------ | --------------- | -------- | -------- |
| MaskRCNN_R50 | MaskRCNN_Head | $\checkmark$       | $C_B \cup C_N$  |          | \        |
| MaskRCNN_R50 | MaskRCNN_Head | $\times$           | $C_B \cup C_N$  |          | \        |
| MaskRCNN_R50 | F-VLM_Head    | $\times$           | $C_B \cup C_N$  |          | \        |
| CLIP_R50     | MaskRCNN_Head | $\times$           | $C_B \cup C_N$  |          | \        |
| CLIP_R50     | F-VLM_Head    | $\times$           | $C_B \cup C_N$  |          | \        |
| CLIP_R50     | F-VLM_Head    | $\times$           | $C_B$           |          |          |

### 2.2 基于LVIS数据集

| Backbone     | Head          | Trainable Backbone | Training source | bbox_mAP | Novel AP |
| ------------ | ------------- | ------------------ | --------------- | -------- | -------- |
| MaskRCNN_R50 | MaskRCNN_Head | $\checkmark$       | $C_B \cup C_N$  |          | \        |
| CLIP_R50     | F-VLM_Head    | $\times$           | $C_B \cup C_N$  |          | \        |
| CLIP_R50     | F-VLM_Head    | $\times$           | $C_B$           |          |          |

## 3. 实验指标

### 3.1 LVIS

根据在训练集中出现的频率将类别分为'frequent'，'common'，'rare'

|               | Category            | Number |
| ------------- | ------------------- | ------ |
| Base Classes  | 'frequent'+'common' | 866    |
| Novel Classes | 'rare'              | 337    |

主要指标：$AP_r$

### 3.2 MSCOCO

|               | Category                                                     | Number |
| ------------- | ------------------------------------------------------------ | ------ |
| Base Classes  | ['person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 'bottle', 'fork', 'spoon', 'bowl','banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'microwave', 'oven', 'toaster','refrigerator', 'book', 'clock', 'vase', 'toothbrush'] | 48     |
| Novel Classes | ['airplane', 'bus', 'cat', 'dog', 'cow', 'elephant', 'umbrella', 'tie', 'snowboard', 'skateboard', 'cup', 'knife', 'cake', 'couch', 'keyboard', 'sink', 'scissors'] | 17     |

#### 主要指标

AP50 of novel categories (Novel AP)

#### 测试命令

1. 测试Novel AP：

```shell
python tools/test.py configs/coco/mask_rcnn_r50_fpn_1x_coco.py ../checkpoints/version_3.0/epoch_7.pth --eval bbox  --eval-options eval_novel=True 
```



2. 测试mAP：



```shell
python tools/test.py configs/coco/mask_rcnn_r50_fpn_1x_coco.py ../checkpoints/version_3.0/epoch_7.pth --eval bbox   
```

