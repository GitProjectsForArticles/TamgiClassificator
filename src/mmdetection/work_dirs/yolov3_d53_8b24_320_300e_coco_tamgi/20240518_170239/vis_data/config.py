backend_args = None
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        0,
        0,
        0,
    ],
    pad_size_divisor=32,
    std=[
        255.0,
        255.0,
        255.0,
    ],
    type="DetDataPreprocessor",
)
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
        depth=53,
        init_cfg=dict(checkpoint="open-mmlab://darknet53", type="Pretrained"),
        out_indices=(
            3,
            4,
            5,
        ),
        type="Darknet",
    ),
    bbox_head=dict(
        anchor_generator=dict(
            base_sizes=[
                [
                    (
                        116,
                        90,
                    ),
                    (
                        156,
                        198,
                    ),
                    (
                        373,
                        326,
                    ),
                ],
                [
                    (
                        30,
                        61,
                    ),
                    (
                        62,
                        45,
                    ),
                    (
                        59,
                        119,
                    ),
                ],
                [
                    (
                        10,
                        13,
                    ),
                    (
                        16,
                        30,
                    ),
                    (
                        33,
                        23,
                    ),
                ],
            ],
            strides=[
                32,
                16,
                8,
            ],
            type="YOLOAnchorGenerator",
        ),
        bbox_coder=dict(type="YOLOBBoxCoder"),
        featmap_strides=[
            32,
            16,
            8,
        ],
        in_channels=[
            512,
            256,
            128,
        ],
        loss_cls=dict(
            loss_weight=1.0, reduction="sum", type="CrossEntropyLoss", use_sigmoid=True
        ),
        loss_conf=dict(
            loss_weight=1.0, reduction="sum", type="CrossEntropyLoss", use_sigmoid=True
        ),
        loss_wh=dict(loss_weight=2.0, reduction="sum", type="MSELoss"),
        loss_xy=dict(
            loss_weight=2.0, reduction="sum", type="CrossEntropyLoss", use_sigmoid=True
        ),
        num_classes=1,
        out_channels=[
            1024,
            512,
            256,
        ],
        type="YOLOV3Head",
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0,
            0,
            0,
        ],
        pad_size_divisor=32,
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type="DetDataPreprocessor",
    ),
    neck=dict(
        in_channels=[
            1024,
            512,
            256,
        ],
        num_scales=3,
        out_channels=[
            512,
            256,
            128,
        ],
        type="YOLOV3Neck",
    ),
    test_cfg=dict(
        conf_thr=0.005,
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.45, type="nms"),
        nms_pre=1000,
        score_thr=0.05,
    ),
    train_cfg=dict(
        assigner=dict(
            min_pos_iou=0, neg_iou_thr=0.5, pos_iou_thr=0.5, type="GridAssigner"
        )
    ),
    type="YOLOV3",
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
    dict(begin=0, by_epoch=False, end=500, start_factor=0.001, type="LinearLR"),
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
        batch_size=8,
        dataset=dict(
            ann_file="labels/coco_annotations_test.json",
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
        ann_file="../../data/dataset/labels/coco_annotations_test.json",
        format_only=False,
        metric=[
            "bbox",
        ],
        type="CocoMetric",
    ),
    type="TestLoop",
)
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file="labels/coco_annotations_test.json",
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
    ann_file="../../data/dataset/labels/coco_annotations_test.json",
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
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=2)
train_dataloader = dict(
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    batch_size=8,
    dataset=dict(
        ann_file="labels/coco_annotations_train.json",
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
        batch_size=8,
        dataset=dict(
            ann_file="labels/coco_annotations_val.json",
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
        ann_file="../../data/dataset/labels/coco_annotations_val.json",
        format_only=False,
        metric=[
            "bbox",
        ],
        type="CocoMetric",
    ),
    type="ValLoop",
)
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file="labels/coco_annotations_val.json",
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
    ann_file="../../data/dataset/labels/coco_annotations_val.json",
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
work_dir = "./work_dirs/yolov3_d53_8b24_320_300e_coco_tamgi"
