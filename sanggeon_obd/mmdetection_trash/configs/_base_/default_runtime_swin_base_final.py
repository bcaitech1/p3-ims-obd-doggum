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
                            project = 'Project4_object_detection',
                            name = 'swin base final')
            )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/opt/ml/sanggeon_obd_2/mmdetection_trash/work_dirs/cascade_mask_rcnn_swin_base_patch4_window7.pth'
# load_from = '/opt/ml/code/mmdetection_trash/checkpoint/cascade_mask_rcnn_swin_tiny_patch4_3x.pth'
# load_from = None
# load_from = './work_dirs/htc_swin_base_patch4_adamw_3x_2/best_bbox_mAP_50.pth'
# resume_from = './work_dirs/htc_swin_base_patch4_adamw_3x_2/best_bbox_mAP_50.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/swin_base_k2_final'