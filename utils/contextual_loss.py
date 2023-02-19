from torch.nn.modules.loss import _Loss as Loss
import torchvision
import torch
from torch.nn import functional as F
from config import opt


class ContextualLoss(Loss):

    def __init__(self, loss='l1', backbone='resnet50', input_channels=1):
        super(ContextualLoss, self).__init__()
        self.backbone = backbone
        self.loss = loss
        self.input_channels = input_channels
        self.net = self.select_backbone()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=opt.device).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=opt.device).view(1,3,1,1))

    def select_backbone(self):
        if self.backbone == 'vgg19':
            net = torchvision.models.vgg19(pretrained=True).features
        elif self.backbone == 'resnet50':
            base = torchvision.models.resnet50(pretrained=True)
            net = torch.nn.Sequential(*list(base.children())[:-3])

        for param in net.parameters():
            param.requires_grad = False


        net = net.to(device=opt.device)
        return net.eval()

    def forward(self, x, y):
        btz = x.size(0)
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = (x-self.mean) / self.std
        y = (y-self.mean) / self.std

        x = self.net(x)
        y = self.net(y)

        x = x.view(btz, -1)
        y = y.view(btz, -1)

        if self.loss == 'l1':
            cx_loss = F.l1_loss(x, y)
        elif self.loss == 'mse':
            cx_loss = F.mse_loss(x, y)
        elif self.loss == 'cros':
            cx_loss = torch.nn.CrossEntropyLoss()(x, y)
        elif self.loss == 'cos':
            x = F.normalize(x, dim=1)
            y = F.normalize(y, dim=1)
            cx_loss = F.cosine_similarity(x, y, dim=-1).mean()
        else:
            cx_loss = F.cross_entropy(x, y)

        return cx_loss