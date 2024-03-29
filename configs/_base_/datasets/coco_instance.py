# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/work/workspace/CV/data/coco/'
img_norm_cfg = dict(
    mean=[0.48145466*255, 0.4578275*255, 0.40821073*255], std=[0.26862954*255, 0.26130258*255, 0.27577711*255], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ('person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat', 'bench', 
           'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
           'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 'bottle', 'fork', 'spoon', 
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
           'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'microwave', 
           'oven', 'toaster', 'refrigerator', 'book', 'clock', 'vase', 'toothbrush')

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
