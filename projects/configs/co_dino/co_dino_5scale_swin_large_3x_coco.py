_base_ = [
    'co_dino_5scale_swin_large_1x_coco.py'
]
pretrained = 'models/co_dino_5scale_swin_large_3x_coco.pth'
# model settings
model = dict(
    backbone=dict(
        drop_path_rate=0.6
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=pretrained
    ),
)

lr_config = dict(policy='step', step=[5])
runner = dict(type='EpochBasedRunner', max_epochs=10)
evaluation = dict(interval=10, metric='bbox')
checkpoint_config = dict(interval=10)
