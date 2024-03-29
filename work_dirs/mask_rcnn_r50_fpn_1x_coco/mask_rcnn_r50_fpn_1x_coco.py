pretrained = '/data/hzf_data/hjj/my_mmdet/RN50.pth'
model = dict(
    type='MaskRCNN',
    backbone=dict(type='clip_image'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHeadTEXT',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=48,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=48,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.3,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
dataset_type = 'CocoDataset'
data_root = '/home/work/workspace/CV/data/coco/'
img_norm_cfg = dict(
    mean=[122.7709383, 116.7460125, 104.09373615000001],
    std=[68.5005327, 66.6321579, 70.32316304999999],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[122.7709383, 116.7460125, 104.09373615000001],
        std=[68.5005327, 66.6321579, 70.32316304999999],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
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
            dict(
                type='Normalize',
                mean=[122.7709383, 116.7460125, 104.09373615000001],
                std=[68.5005327, 66.6321579, 70.32316304999999],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
classes = ('person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat',
           'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe',
           'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'kite',
           'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut',
           'chair', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
           'microwave', 'oven', 'toaster', 'refrigerator', 'book', 'clock',
           'vase', 'toothbrush')
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        classes=('person', 'bicycle', 'car', 'motorcycle', 'train', 'truck',
                 'boat', 'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra',
                 'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee',
                 'skis', 'kite', 'surfboard', 'bottle', 'fork', 'spoon',
                 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                 'carrot', 'pizza', 'donut', 'chair', 'bed', 'toilet', 'tv',
                 'laptop', 'mouse', 'remote', 'microwave', 'oven', 'toaster',
                 'refrigerator', 'book', 'clock', 'vase', 'toothbrush'),
        type='CocoDataset',
        ann_file=
        '/home/work/workspace/CV/data/coco/annotations/instances_train2017.json',
        img_prefix='/home/work/workspace/CV/data/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[122.7709383, 116.7460125, 104.09373615000001],
                std=[68.5005327, 66.6321579, 70.32316304999999],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(
        classes=('person', 'bicycle', 'car', 'motorcycle', 'train', 'truck',
                 'boat', 'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra',
                 'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee',
                 'skis', 'kite', 'surfboard', 'bottle', 'fork', 'spoon',
                 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                 'carrot', 'pizza', 'donut', 'chair', 'bed', 'toilet', 'tv',
                 'laptop', 'mouse', 'remote', 'microwave', 'oven', 'toaster',
                 'refrigerator', 'book', 'clock', 'vase', 'toothbrush'),
        type='CocoDataset',
        ann_file=
        '/home/work/workspace/CV/data/coco/annotations/instances_val2017.json',
        img_prefix='/home/work/workspace/CV/data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[122.7709383, 116.7460125, 104.09373615000001],
                        std=[68.5005327, 66.6321579, 70.32316304999999],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        classes=('person', 'bicycle', 'car', 'motorcycle', 'train', 'truck',
                 'boat', 'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra',
                 'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee',
                 'skis', 'kite', 'surfboard', 'bottle', 'fork', 'spoon',
                 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                 'carrot', 'pizza', 'donut', 'chair', 'bed', 'toilet', 'tv',
                 'laptop', 'mouse', 'remote', 'microwave', 'oven', 'toaster',
                 'refrigerator', 'book', 'clock', 'vase', 'toothbrush'),
        type='CocoDataset',
        ann_file=
        '/home/work/workspace/CV/data/coco/annotations/instances_val2017.json',
        img_prefix='/home/work/workspace/CV/data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[122.7709383, 116.7460125, 104.09373615000001],
                        std=[68.5005327, 66.6321579, 70.32316304999999],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(metric=['bbox', 'segm'])
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
work_dir = './work_dirs/mask_rcnn_r50_fpn_1x_coco'
auto_resume = False
gpu_ids = [0]
