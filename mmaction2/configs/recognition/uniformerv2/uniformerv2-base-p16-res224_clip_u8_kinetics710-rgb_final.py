ann_file_test = '/home/minh/Documents/surg2024/data/data_new/annotations/test_video.txt'
ann_file_train = '/home/minh/Documents/surg2024/data/data_new/annotations/fold_4/train_video.txt'
ann_file_val = '/home/minh/Documents/surg2024/data/data_new/annotations/fold_4/val_video.txt'
auto_scale_lr = dict(base_batch_size=256, enable=False)
base_lr = 2e-05
data_root = '/home/minh/Documents/surg2024/data/data_new/data_1fps'
data_root_val = '/home/minh/Documents/surg2024/data/data_new/data_1fps'
dataset_type = 'VideoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=3, max_keep_ckpts=3, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=40, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        backbone_drop_path_rate=0.0,
        clip_pretrained=False,
        double_lmhra=True,
        drop_path_rate=0.0,
        dw_reduction=1.5,
        heads=12,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-pre_u8_kinetics710-rgb/uniformerv2-base-p16-res224_clip-pre_u8_kinetics710-rgb_20230612-63cdbad9.pth',
            prefix='backbone.',
            type='Pretrained'),
        input_resolution=224,
        layers=12,
        mlp_dropout=[
            0.5,
            0.5,
            0.5,
            0.5,
        ],
        mlp_factor=4.0,
        n_dim=768,
        n_head=12,
        n_layers=4,
        no_lmhra=True,
        patch_size=16,
        return_list=[
            8,
            9,
            10,
            11,
        ],
        t_size=8,
        temporal_downsample=False,
        type='UniFormerV2',
        width=768),
    cls_head=dict(
        average_clips='prob',
        channel_map=
        'configs/recognition/uniformerv2/k710_channel_map/surgvu2024.json',
        dropout_ratio=0.5,
        in_channels=768,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-pre_u8_kinetics710-rgb/uniformerv2-base-p16-res224_clip-pre_u8_kinetics710-rgb_20230612-63cdbad9.pth',
            prefix='cls_head.',
            type='Pretrained'),
        num_classes=7,
        type='UniFormerHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            114.75,
            114.75,
            114.75,
        ],
        std=[
            57.375,
            57.375,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
num_frames = 8
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=2e-05, type='AdamW', weight_decay=0.005),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.5,
        type='LinearLR'),
    dict(
        T_max=50,
        begin=5,
        by_epoch=True,
        convert_to_iter_based=True,
        end=50,
        eta_min_ratio=0.1,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=
        '/home/minh/Documents/surg2024/data/data_new/annotations/test_video.txt',
        data_prefix=dict(
            video='/home/minh/Documents/surg2024/data/data_new/data_1fps'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=8, num_clips=4, test_mode=True, type='UniformSample'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                224,
            ), type='Resize'),
            dict(crop_size=224, type='ThreeCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=8, num_clips=4, test_mode=True, type='UniformSample'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        224,
    ), type='Resize'),
    dict(crop_size=224, type='ThreeCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=50, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        '/home/minh/Documents/surg2024/data/data_new/annotations/fold_4/train_video.txt',
        data_prefix=dict(
            video='/home/minh/Documents/surg2024/data/data_new/data_1fps'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(clip_len=8, num_clips=1, type='UniformSample'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(
                magnitude=7,
                num_layers=4,
                op='RandAugment',
                type='PytorchVideoWrapper'),
            dict(type='RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='VideoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=8, num_clips=1, type='UniformSample'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(
        magnitude=7,
        num_layers=4,
        op='RandAugment',
        type='PytorchVideoWrapper'),
    dict(type='RandomResizedCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        '/home/minh/Documents/surg2024/data/data_new/annotations/fold_4/val_video.txt',
        data_prefix=dict(
            video='/home/minh/Documents/surg2024/data/data_new/data_1fps'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=8, num_clips=1, test_mode=True, type='UniformSample'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                224,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='AccMetric')
val_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=8, num_clips=1, test_mode=True, type='UniformSample'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        224,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/uniformerv2-base-p16-res224_clip_u8_kinetics710-rgb'
