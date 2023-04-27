_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

load_from = "/data/hzf_data/hjj/epoch_7.pth"
# optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.000025)
# evaluation = dict(type="SubModulesDistEvalHook", interval=4000)
# evaluation = dict(interval=2,metric=['bbox', 'segm'])
# model = dict(
#     backbone=dict(
#         type = 'Resnet50',
#         name = "RN50"
#     ),
#     neck=dict(
#         type='FPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         norm_cfg=dict(type='SyncBN', requires_grad=True),
#         num_outs=5),
#     roi_head=dict(
#         bbox_head=dict(
#             type='Shared4Conv1FCBBoxHead',
#             in_channels=256,
#             ensemble=False,
#             fc_out_channels=1024,
#             roi_feat_size=7,
#             with_cls=False,
#             num_classes=80,
#             norm_cfg=dict(type='SyncBN', requires_grad=True),
#             bbox_coder=dict(
#                 type='DeltaXYWHBBoxCoder',
#                 target_means=[0., 0., 0., 0.],
#                 target_stds=[0.1, 0.1, 0.2, 0.2]),
#             reg_class_agnostic=True,
#             loss_cls=dict(
#                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#             loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
#         mask_head=dict(num_classes=80)))
