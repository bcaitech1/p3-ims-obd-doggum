from .segnet import SegNetVgg16


def get_model(args, num_classes):
    if args.model == 'segnetvgg16':
        model = SegNetVgg16(num_classes=num_classes)

    return model