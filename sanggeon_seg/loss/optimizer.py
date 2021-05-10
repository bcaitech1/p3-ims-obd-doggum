import torch

def get_optimizer(args, net):
    """
    Decide Optimizer
    :param args:
    :return:
    """
    param_groups = net.parameters()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=param_groups, lr=args.lr, weight_decay=1e-6)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=param_groups, lr=args.lr, weight_decay=1e-6)

    for g in optimizer.param_groups:
        g['lr'] = g['lr'] / 3

    return optimizer