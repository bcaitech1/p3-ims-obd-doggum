from .segnet import SegNetVgg16
import segmentation_models_pytorch as smp

def get_model(args, num_classes):
    if args.model == 'segnetvgg16':
        model = SegNetVgg16(num_classes=num_classes)
    if args.model == 'unetefb0':
        model = smp.Unet(encoder_name='efficientnet-b0', classes=num_classes, encoder_weights='imagenet', activation=None)
    if args.model == 'unetmnv2':
        model = smp.Unet(encoder_name='mobilenet_v2', classes=num_classes, encoder_weights='imagenet', activation=None)
    if args.model == 'deeplabv3mnv2':
        model = smp.DeepLabV3Plus(encoder_name='mobilenet_v2', classes=num_classes, encoder_weights='imagenet', activation=None)
    if args.model == 'deeplabv3plus-se_resnext50':
        model = smp.DeepLabV3Plus(encoder_name='se_resnext50_32x4d', classes=num_classes, encoder_weights='imagenet', activation=None)
    if args.model == 'deeplabv3plus-se_resnext101':
        model = smp.DeepLabV3Plus(encoder_name='se_resnext101_32x4d', classes=num_classes, encoder_weights='imagenet',
                                 activation=None)

    # ValueError: InceptionResnetV2 encoder does not support dilated mode due to pooling operation for downsampling!
    if args.model == 'deeplabv3plus-inceptionresnetv2':
        model = smp.DeepLabV3Plus(encoder_name='inceptionresnetv2', classes=num_classes, encoder_weights='imagenet+background', activation=None)

    # ValueError: InceptionResnetV2 encoder does not support dilated mode due to pooling operation for downsampling!
    if args.model == 'deeplabv3plus-inceptionv4':
        model = smp.DeepLabV3Plus(encoder_name='inceptionv4', classes=num_classes, encoder_weights='imagenet+background', activation=None)
    # Too big.
    if args.model == 'unetplusplus-inceptionv4':
        model = smp.UnetPlusPlus(encoder_name='inceptionresnetv2', classes=num_classes, encoder_weights='imagenet+background', activation=None)

    if args.model == 'deeplabv3plus-timm-regnety_120':
        model = smp.DeepLabV3Plus(encoder_name='timm-regnety_120', classes=num_classes, encoder_weights='imagenet', activation=None)

    if args.model == 'deeplabv3plus-mobilenet_v2':
        model = smp.DeepLabV3Plus(encoder_name='mobilenet_v2', classes=num_classes, encoder_weights='imagenet',
                                  activation=None)

    return model