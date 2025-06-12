auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = '/kfs2/projects/pvfleets24/repos/cv-dl-framework'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        rule='greater', save_best='coco/bbox_mAP_50', type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/kfs2/projects/pvfleets24/repos/cv-dl-framework/runs/18_05_2025_13_16_01/best_coco_bbox_mAP_50_epoch_18.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(classes=('panel', ))
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=None,
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='caffe',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        pad_size_divisor=32,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.04,
                    0.04,
                    0.08,
                    0.08,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
            loss_cls=dict(
                loss_weight=1.5, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        test_cfg=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.001),
        train_cfg=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.65,
                neg_iou_thr=0.65,
                pos_iou_thr=0.65,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.25,
                type='RandomSampler')),
        type='StandardRoIHead'),
    rpn_head=dict(
        num_stages=2,
        stages=[
            dict(
                adapt_cfg=dict(dilation=3, type='dilation'),
                anchor_generator=dict(
                    ratios=[
                        1.0,
                    ],
                    scales=[
                        8,
                    ],
                    strides=[
                        4,
                        8,
                        16,
                        32,
                        64,
                    ],
                    type='AnchorGenerator'),
                bbox_coder=dict(
                    target_means=(
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ),
                    target_stds=(
                        0.1,
                        0.1,
                        0.5,
                        0.5,
                    ),
                    type='DeltaXYWHBBoxCoder'),
                bridged_feature=True,
                feat_channels=256,
                in_channels=256,
                loss_bbox=dict(linear=True, loss_weight=7.0, type='IoULoss'),
                reg_decoded_bbox=True,
                test_cfg=dict(
                    max_per_img=300,
                    min_bbox_size=0,
                    nms=dict(iou_threshold=0.8, type='nms'),
                    nms_pre=1000),
                train_cfg=dict(
                    allowed_border=-1,
                    assigner=dict(
                        center_ratio=0.2,
                        ignore_ratio=0.5,
                        type='RegionAssigner'),
                    debug=False,
                    pos_weight=-1),
                type='StageCascadeRPNHead',
                with_cls=False),
            dict(
                adapt_cfg=dict(type='offset'),
                bbox_coder=dict(
                    target_means=(
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ),
                    target_stds=(
                        0.05,
                        0.05,
                        0.1,
                        0.1,
                    ),
                    type='DeltaXYWHBBoxCoder'),
                bridged_feature=False,
                feat_channels=256,
                in_channels=256,
                loss_bbox=dict(linear=True, loss_weight=7.0, type='IoULoss'),
                loss_cls=dict(
                    loss_weight=0.7, type='CrossEntropyLoss',
                    use_sigmoid=True),
                reg_decoded_bbox=True,
                test_cfg=dict(
                    max_per_img=300,
                    min_bbox_size=0,
                    nms=dict(iou_threshold=0.8, type='nms'),
                    nms_pre=1000),
                train_cfg=dict(
                    allowed_border=-1,
                    assigner=dict(
                        ignore_iof_thr=-1,
                        min_pos_iou=0.3,
                        neg_iou_thr=0.7,
                        pos_iou_thr=0.7,
                        type='MaxIoUAssigner'),
                    debug=False,
                    pos_weight=-1,
                    sampler=dict(
                        add_gt_as_proposals=False,
                        neg_pos_ub=-1,
                        num=256,
                        pos_fraction=0.5,
                        type='RandomSampler')),
                type='StageCascadeRPNHead',
                with_cls=True),
        ],
        type='CascadeRPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.001),
        rpn=dict(
            max_per_img=300,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.8, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.65,
                neg_iou_thr=0.65,
                pos_iou_thr=0.65,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=[
            dict(
                allowed_border=-1,
                assigner=dict(
                    center_ratio=0.2, ignore_ratio=0.5, type='RegionAssigner'),
                debug=False,
                pos_weight=-1),
            dict(
                allowed_border=-1,
                assigner=dict(
                    ignore_iof_thr=-1,
                    min_pos_iou=0.3,
                    neg_iou_thr=0.7,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=False,
                    neg_pos_ub=-1,
                    num=256,
                    pos_fraction=0.5,
                    type='RandomSampler')),
        ],
        rpn_proposal=dict(
            max_per_img=300,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.8, type='nms'),
            nms_pre=2000)),
    type='FasterRCNN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
rpn_weight = 0.7
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/kfs2/projects/pvfleets24/repos/cv-dl-framework/test/label_json.json',
        backend_args=None,
        data_prefix=dict(
            img='/kfs2/projects/pvfleets24/repos/cv-dl-framework/test/images/'
        ),
        data_root='/kfs2/projects/pvfleets24/repos/cv-dl-framework',
        metainfo=dict(classes=('panel', )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/kfs2/projects/pvfleets24/repos/cv-dl-framework/test/label_json.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=300, type='EpochBasedTrainLoop', val_interval=3)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file=
        '/kfs2/projects/pvfleets24/repos/cv-dl-framework/train/label_json.json',
        backend_args=None,
        data_prefix=dict(
            img='/kfs2/projects/pvfleets24/repos/cv-dl-framework/train/images/'
        ),
        data_root='/kfs2/projects/pvfleets24/repos/cv-dl-framework',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=('panel', )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/kfs2/projects/pvfleets24/repos/cv-dl-framework/test/label_json.json',
        backend_args=None,
        data_prefix=dict(
            img='/kfs2/projects/pvfleets24/repos/cv-dl-framework/test/images/'
        ),
        data_root='/kfs2/projects/pvfleets24/repos/cv-dl-framework',
        metainfo=dict(classes=('panel', )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/kfs2/projects/pvfleets24/repos/cv-dl-framework/test/label_json.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(
            save_dir=
            '/kfs2/projects/pvfleets24/repos/cv-dl-framework/runs/18_05_2025_13_16_01',
            type='LocalVisBackend'),
    ])
work_dir = '/kfs2/projects/pvfleets24/repos/cv-dl-framework/runs/18_05_2025_13_16_01'
