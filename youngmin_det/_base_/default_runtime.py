checkpoint_config = dict(max_keep_ckpts = 2, interval=1)
evaluation = dict(
    interval = 1,
    save_best='bbox_mAP_50',
    metric='bbox',
    rule = 'greater'
)
# yapf:disable

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook', 
            init_kwargs = dict(
                            project = 'Pstage4_object_detection',
                            name = 'cascade_mask_swin_kfold4_cosine_restart')
        )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]
work_dirs = '/opt/ml/code/mmdetection_trash/work_dirs'
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/opt/ml/code/mmdetection_trash/Swin-Transformer-Object-Detection/configs/trash/swin_transformer/cascade_mask_rcnn_swin_base_patch4_window7.pth'
resume_from = '/opt/ml/code/mmdetection_trash/Swin-Transformer-Object-Detection/work_dirs/cascade_mask_swin/epoch_3.pth'
# resume_from = '/opt/ml/code/mmdetection_trash/Swin-Transformer-Object-Detection/work_dirs/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/epoch_11.pth'
workflow = [('train', 1)]
