from .segnet import SegNetVgg16
import segmentation_models_pytorch as smp

def get_model(args, num_classes):
    if args.model == 'segnetvgg16':
        model = SegNetVgg16(num_classes=num_classes)
    if args.model == 'unetefb0':
        model = smp.Unet(encoder_name='efficientnet-b0', classes=num_classes, encoder_weights='imagenet', activation=None)
    if args.model == 'unetmnv2':
        model = smp.Unet(encoder_name='mobilenet_v2', classes=num_classes, encoder_weights='imagenet', activation=None)

    return model