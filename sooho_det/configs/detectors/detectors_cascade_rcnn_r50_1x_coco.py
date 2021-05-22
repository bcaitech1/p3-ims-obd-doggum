_base_ = [
    '../../_base_/models/cascade_rcnn_r50_fpn.py',
    '../dataset.py',
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='DetectoRS_ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        pretrained='open-mmlab://resnext101_32x4d',
        conv_cfg=dict(type='ConvAWS'),
        norm_cfg=dict(type='BN', requires_grad=True),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNeXt',
            depth=101,
            groups=32,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='open-mmlab://resnext101_32x4d',
            style='pytorch')))
