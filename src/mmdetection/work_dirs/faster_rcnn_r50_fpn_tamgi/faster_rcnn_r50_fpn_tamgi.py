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
load_from = "/home/ana/University/Tamgi/src/mmdetection/work_dirs/faster_rcnn_r50_fpn_tamgi/best_coco_bbox_mAP_epoch_30.pth"
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
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type="FPN",
    ),
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
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type="DeltaXYWHBBoxCoder",
            ),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type="L1Loss"),
            loss_cls=dict(loss_weight=1.0, type="CrossEntropyLoss", use_sigmoid=False),
            num_classes=80,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type="Shared2FCBBoxHead",
        ),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type="RoIAlign"),
            type="SingleRoIExtractor",
        ),
        type="StandardRoIHead",
    ),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
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
        loss_cls=dict(loss_weight=1.0, type="CrossEntropyLoss", use_sigmoid=True),
        type="RPNHead",
    ),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100, nms=dict(iou_threshold=0.5, type="nms"), score_thr=0.05
        ),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type="nms"),
            nms_pre=1000,
        ),
    ),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type="MaxIoUAssigner",
            ),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type="RandomSampler",
            ),
        ),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type="MaxIoUAssigner",
            ),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type="RandomSampler",
            ),
        ),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type="nms"),
            nms_pre=2000,
        ),
    ),
    type="FasterRCNN",
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
        batch_size=12,
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
    batch_size=12,
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
    batch_size=12,
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
        batch_size=12,
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
    batch_size=12,
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
work_dir = "./work_dirs/faster_rcnn_r50_fpn_tamgi"
