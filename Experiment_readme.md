#  实验记录

## 1. 实验

### 1.1 基于LVIS数据集：

### 1.2 基于MSCOCO数据集：

#### 1.2.1 数据集划分：

![Screenshot 2023-03-16 14:11:30](/home/qian/Pictures/Screenshot 2023-03-16 14:11:30.png)

参照detpro，将coco划分为base（48类）和novel（17类）其余为不可见类（15类）

#### 1.2.2 实验结果：

| 实验序号        | 描述                                            | bbox_mAP |
| --------------- | ----------------------------------------------- | -------- |
| f_vlm_coco_v1.0 | 修改$\alpha$=0.2，$\beta$=0.45；修改softmax位置 |          |
| f_vlm_coco_v2.0 |                                                 |          |
| f_vlm_coco_v3.0 |                                                 |          |



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



