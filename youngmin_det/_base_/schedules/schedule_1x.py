# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type = 'Adam', lr = 0.0001, weight_decay = 0.00001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[])
    #step = [5, 10, 15, 20, 25]로 해보기
runner = dict(type='EpochBasedRunner', max_epochs=20)

