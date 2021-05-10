import torch.nn as nn

def get_loss(args):
    """
    get the criterion based on loss function
    :param args:
    :return:
    """

    if args.criterion == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()

    return criterion