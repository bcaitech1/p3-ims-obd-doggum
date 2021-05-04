import torch
import torch.nn.functional as F

def to_one_hot(tensor, nClasses, device):
    n, h, w = tensor.size()
    one_hot = torch.zeros(n, nClasses, h, w).to(device).scatter_(1, tensor.view(n, 1, h, w), 1)
    return one_hot


class mIoULoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=4, device='cuda'):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.device = device

    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W
        inputs = inputs.to(self.device)
        target = target.to(self.device)

        SMOOTH = 1e-6
        N = inputs.size()[0]

        inputs = F.softmax(inputs, dim=1)
        target_oneHot = to_one_hot(target, self.classes, self.device)
        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2) + SMOOTH

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2) + SMOOTH

        loss = inter / union

        ## Return average loss over classes and batch
        return -loss.mean()