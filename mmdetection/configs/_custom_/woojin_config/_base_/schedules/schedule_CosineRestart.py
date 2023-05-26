# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None) # in boostcamp default : dict(grad_clip=dict(max_keep_ckpts=3, interval=1))  
# learning policy
lr_config = dict(
    policy='CosineRestart', 
    periods=[3900 for _ in range(18)],
    restart_weights=[1],
    by_epoch = True,
    min_lr=1e-07)
runner = dict(type='EpochBasedRunner', max_epochs=20)