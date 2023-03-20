# F-vlm修改日志

## 1. 冻结Backbone注意

### 1.1 关于BN层冻结

因为内存限制，batch size比较小，在batch size比较小的情况下用了BN层反而会使模型稳定性降低。如今很多网络都是在imagenet上pre-trained的，已经可以一定程度上达到bn层本身的三个作用点，所以这时可以freeze BN层。 这也是faster-rcnn和其他popular网络的做法。

加载预训练模型时，如果只将 para.requires_grad = False ，并不能完全冻结模型的参数，因为模型中的 BN 层并不随 loss.backward() 与 optimizer.step() 来更新，而是在模型 forward 的过程中基于动量来更新，因此需要每个 forward 之前冻结 BN 层：

BN层的统计数据更新是在每一次训练阶段model.train()后的forward()方法中自动实现的

《MMDetection: Open MMLab Detection Toolbox and Benchmark》里面的相关实验，在mmdetection中eval = True, requires grad = True是默认设置，不更新BN层的统计信息，也就是running_var和running_mean，但是优化更新其weight和bias的学习参数。


![在这里插入图片描述](https://img-blog.csdnimg.cn/e3ccfcea098f40bc87ad28ed5ecca481.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5LuZ5aWz5L-u54K85Y-y,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

## 2. ZSD eval metric

### 2.1 参考DetPro中coco.py修改

```python
self.ignore_ids_train = [4, 5, 9, 10, 11, 12, 15, 16, 19, 20, 25, 27, 31, 32, 34, 35, 36, 38, 40, 41, 43, 52, 55, 57, 58, 60, 66, 67, 71, 76, 77, 78]
self.ignore_ids_test = [9, 10, 11, 12, 32, 34, 35, 38, 40, 52, 58, 60, 67, 77, 78]
CLASSES_for_evaluate = []
if self.test_mode:
	for id in range(len(self.CLASSES)):
    	if id not in self.ignore_ids_test:          						CLASSES_for_evaluate.append(self.CLASSES[id])
    self.cat_ids_for_evaluate = self.coco.get_cat_ids(cat_names=CLASSES_for_evaluate)
```

### 2.2 mertic对齐

![image-20230318114546922](/home/qian/.config/Typora/typora-user-images/image-20230318114546922.png)

![image-20230318114612618](/home/qian/.config/Typora/typora-user-images/image-20230318114612618.png)

![image-20230318114709469](/home/qian/.config/Typora/typora-user-images/image-20230318114709469.png)

结论: F-VLM的指标是在Generalized（17+18）下的Novel AP以及mAP

> 注：表中出现的另外两列指标（Open-Vocabulary Object Detection Using Captions）都是基于Base类别的数据集进行训练；
>
> 分别替换classifier head为Base类embedding和Novel类embedding；（在single_test里修改cls_score）
>
> 分别使用基类标注，新类别标注进行evaluate计算，得到表格中COCO前两列的指标（Base48，Novel17）；
>
> 分别对应全监督和零样本检测；

#### 2.2.1 问题1 ：OVOD与ZSD区别

具体见coco.py中self.ignore_cats:

```python
# OVOD 开放词汇目标检测
for cat in self.coco.cats:
                # if self.cat2label[cat] not in self.ignore_ids_train:
                if self.cat2label[cat] not in self.ignore_ids_test:
                    cats[cat] = self.coco.cats[cat]
                else:
                    self.ignore_cats.append(cat)
```

![image-20230320104805774](/home/qian/.config/Typora/typora-user-images/image-20230320104805774.png)

```python
# ZSD 零样本检测
for cat in self.coco.cats:
                if self.cat2label[cat] not in self.ignore_ids_train:
                # if self.cat2label[cat] not in self.ignore_ids_test:
                    cats[cat] = self.coco.cats[cat]
                else:
                    self.ignore_cats.append(cat)
```

![image-20230320105733368](/home/qian/.config/Typora/typora-user-images/image-20230320105733368.png)

影响训练时的loss大小：

![image-20230320191015952](/home/qian/.config/Typora/typora-user-images/image-20230320191015952.png)





