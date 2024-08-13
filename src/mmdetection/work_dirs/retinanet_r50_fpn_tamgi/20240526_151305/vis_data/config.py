backend_args = None
data_root = "../../data/dataset/"
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=3, save_best="coco/bbox_mAP", type="CheckpointHook"
    ),
    logger=dict(interval=50, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(type="DetVisualizationHook"),
)
default_scope = "mmdet"
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)
launcher = "none"
load_from = None
log_level = "INFO"
log_processor = dict(by_epoch=True, type="LogProcessor", window_size=50)
metainfo = dict(
    classes="tamgi",
    palette=[
        (
            220,
            20,
            60,
        ),
    ],
)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint="torchvision://resnet50", type="Pretrained"),
        norm_cfg=dict(requires_grad=True, type="BN"),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style="pytorch",
        type="ResNet",
    ),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type="AnchorGenerator",
        ),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type="DeltaXYWHBBoxCoder",
        ),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type="L1Loss"),
        num_classes=1,
        stacked_convs=4,
        type="RetinaHead",
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type="DetDataPreprocessor",
    ),
    neck=dict(
        add_extra_convs="on_input",
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type="FPN",
    ),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.5, type="nms"),
        nms_pre=1000,
        score_thr=0.05,
    ),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.4,
            pos_iou_thr=0.5,
            type="MaxIoUAssigner",
        ),
        debug=False,
        pos_weight=-1,
        sampler=dict(type="PseudoSampler"),
    ),
    type="RetinaNet",
)
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type="Adam", weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(decay_mult=1.0, lr_mult=0.1), head=dict(lr_mult=1.0)
        )
    ),
)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type="MultiStepLR",
    ),
]
resume = False
test_cfg = dict(
    dataloader=dict(
        batch_size=12,
        dataset=dict(
            ann_file="detection_labels/coco_annotations_test.json",
            backend_args=None,
            data_prefix=dict(img=""),
            data_root="../../data/dataset/",
            metainfo=dict(
                classes="tamgi",
                palette=[
                    (
                        220,
                        20,
                        60,
                    ),
                ],
            ),
            pipeline=[
                dict(backend_args=None, type="LoadImageFromFile"),
                dict(
                    keep_ratio=True,
                    scale=(
                        1333,
                        800,
                    ),
                    type="Resize",
                ),
                dict(
                    meta_keys=(
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                    ),
                    type="PackDetInputs",
                ),
            ],
            test_mode=True,
            type="CocoDataset",
        ),
        drop_last=False,
        num_workers=8,
        persistent_workers=True,
        sampler=dict(shuffle=False, type="DefaultSampler"),
    ),
    evaluator=dict(
        ann_file="../../data/dataset/detection_labels/coco_annotations_test.json",
        format_only=False,
        metric=[
            "bbox",
        ],
        type="CocoMetric",
    ),
    type="TestLoop",
)
test_dataloader = dict(
    batch_size=12,
    dataset=dict(
        ann_file="detection_labels/coco_annotations_test.json",
        backend_args=None,
        data_prefix=dict(img=""),
        data_root="../../data/dataset/",
        metainfo=dict(
            classes="tamgi",
            palette=[
                (
                    220,
                    20,
                    60,
                ),
            ],
        ),
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type="Resize",
            ),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        type="CocoDataset",
    ),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
test_evaluator = dict(
    ann_file="../../data/dataset/detection_labels/coco_annotations_test.json",
    format_only=False,
    metric=[
        "bbox",
    ],
    type="CocoMetric",
)
test_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(
        keep_ratio=True,
        scale=(
            1333,
            800,
        ),
        type="Resize",
    ),
    dict(
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
        type="PackDetInputs",
    ),
]
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=2)
train_dataloader = dict(
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    batch_size=10,
    dataset=dict(
        ann_file="detection_labels/coco_annotations_train.json",
        backend_args=None,
        data_prefix=dict(img=""),
        data_root="../../data/dataset/",
        metainfo=dict(
            classes="tamgi",
            palette=[
                (
                    220,
                    20,
                    60,
                ),
            ],
        ),
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                poly2mask=False, type="LoadAnnotations", with_bbox=True, with_mask=False
            ),
            dict(keep_ratio=True, scale=(960,), type="Resize"),
            dict(prob=0.5, type="RandomFlip"),
            dict(type="PackDetInputs"),
        ],
        type="CocoDataset",
    ),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)
train_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(poly2mask=False, type="LoadAnnotations", with_bbox=True, with_mask=False),
    dict(keep_ratio=True, scale=(960,), type="Resize"),
    dict(prob=0.5, type="RandomFlip"),
    dict(type="PackDetInputs"),
]
val_cfg = dict(
    dataloader=dict(
        batch_size=12,
        dataset=dict(
            ann_file="detection_labels/coco_annotations_val.json",
            backend_args=None,
            data_prefix=dict(img=""),
            data_root="../../data/dataset/",
            metainfo=dict(
                classes="tamgi",
                palette=[
                    (
                        220,
                        20,
                        60,
                    ),
                ],
            ),
            pipeline=[
                dict(backend_args=None, type="LoadImageFromFile"),
                dict(
                    keep_ratio=True,
                    scale=(
                        1333,
                        800,
                    ),
                    type="Resize",
                ),
                dict(
                    meta_keys=(
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                    ),
                    type="PackDetInputs",
                ),
            ],
            test_mode=True,
            type="CocoDataset",
        ),
        drop_last=False,
        num_workers=8,
        persistent_workers=True,
        sampler=dict(shuffle=False, type="DefaultSampler"),
    ),
    evaluator=dict(
        ann_file="../../data/dataset/detection_labels/coco_annotations_val.json",
        format_only=False,
        metric=[
            "bbox",
        ],
        type="CocoMetric",
    ),
    type="ValLoop",
)
val_dataloader = dict(
    batch_size=12,
    dataset=dict(
        ann_file="detection_labels/coco_annotations_val.json",
        backend_args=None,
        data_prefix=dict(img=""),
        data_root="../../data/dataset/",
        metainfo=dict(
            classes="tamgi",
            palette=[
                (
                    220,
                    20,
                    60,
                ),
            ],
        ),
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type="Resize",
            ),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        type="CocoDataset",
    ),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
val_evaluator = dict(
    ann_file="../../data/dataset/detection_labels/coco_annotations_val.json",
    format_only=False,
    metric=[
        "bbox",
    ],
    type="CocoMetric",
)
vis_backends = [
    dict(type="LocalVisBackend"),
]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
    ],
)
work_dir = "./work_dirs/retinanet_r50_fpn_tamgi"
