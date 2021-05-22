optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    periods=[20,15,10,5,5,5],
    restart_weights = [1, 0.25, 0.2,0.15,0.1, 0.05],
    min_lr_ratio=1e-6)
runner = dict(type='EpochBasedRunnerAmp', max_epochs=60)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)