checkpoint_config = dict(max_keep_ckpts=1, interval=1)
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook', 
            init_kwargs = dict(
                            project = 'Project3_object_detection',
                            name = 'htc_swin_base_last_5x_continue')
            )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = '/opt/ml/code/mmdetection_trash/checkpoint/cascade_mask_rcnn_swin_base_patch4.pth'
# load_from = '/opt/ml/code/mmdetection_trash/checkpoint/cascade_mask_rcnn_swin_tiny_patch4_3x.pth'
# load_from = None
load_from = None

# resume_from = './work_dirs/htc_swin_base_patch4_adamw_3x_2/best_bbox_mAP_50.pth'
# resume_from = 'work_dirs/htc_swin_base_patch4_adamw_3x_v5_last/epoch_36.pth'
resume_from = '/opt/ml/code/mmdetection_trash/work_dirs/htc_swin_base_last_5x_continue/epoch_42.pth'
workflow = [('train', 1)]
work_dir = './work_dirs/htc_swin_base_last_5x_continue'