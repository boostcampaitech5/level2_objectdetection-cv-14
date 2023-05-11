# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.00001)
optimizer_config = dict(grad_clip=None)
# learning policy

# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[16, 22])

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1 / 10,
    min_lr=1e-6)

    # Use a linear warm-up at [0, 100) iterations
    
runner = dict(type='EpochBasedRunner', max_epochs=20)
