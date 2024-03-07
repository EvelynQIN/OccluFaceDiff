norm_cfg = dict(type="SyncBN", requires_grad=True)

data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512),
    std=[58.395, 57.12, 57.375],
    type="SegDataPreProcessor",
)

test_pipeline = [
    dict(type="LoadImageFromNDArray"),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="PackSegInputs"),
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_model = dict(type="SegTTAModel")
tta_pipeline = [
    dict(type="LoadImageFromNDArray", backend_args=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [dict(type="Resize", scale_factor=r, keep_ratio=True) for r in img_ratios],
            [
                dict(type="RandomFlip", prob=0.0, direction="horizontal"),
                dict(type="RandomFlip", prob=1.0, direction="horizontal"),
            ],
            [dict(type="LoadAnnotations")],
            [dict(type="PackSegInputs")],
        ],
    ),
]

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained="open-mmlab://resnet101_v1c",
    backbone=dict(
        type="ResNetV1c",
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style="pytorch",
        contract_dilation=True,
    ),
    decode_head=dict(
        type="DepthwiseSeparableASPPHead",
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        sampler=dict(type="OHEMPixelSampler", thresh=0.7, min_kept=10000),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
