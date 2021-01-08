import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH



class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3): # 64, 2
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0) # 10 means 5 landmarks

    def forward(self,x):
        # print(x.shape) # [32, 64, 80, 80]
        out = self.conv1x1(x) # num of channels decrease to 20, which directly maps to 80*80 every pixel have 2 anchors, each one predicts 10 values for landmark
        # print(out.shape) # [32, 20, 80, 80]
        out = out.permute(0,2,3,1).contiguous()
        # print(out.shape) # [32, 80, 80, 20]
        # a=out.view(out.shape[0], -1, 10) # -1 means: 32*80*80*20 / 32*10 == 12800?
        # print(a.shape) # [32, 12800, 10]
        return out.view(out.shape[0], -1, 10)

class GazeHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(GazeHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 2)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("../../../data/checkpoints/Face/retinaface/weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel']) # 64
        self.GazeHead = self._make_gaze_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2): # exact param
        # input [32, 64, 80, 80], output [32, 12800, 10]
        # test = self.LandmarkHead[0](feature1)
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num)) # 64,2
        return landmarkhead

    def _make_gaze_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        gazehead = nn.ModuleList()
        for i in range(fpn_num):
            gazehead.append(GazeHead(inchannels,anchor_num))
        return gazehead

    def forward(self,inputs):
        # inputs.shape == torch.Size([32, 3, 640, 640])
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out) 
        # fpn[0].shape == torch.Size([32, 64, 80, 80])
        # fpn[1].shape == torch.Size([32, 64, 40, 40])
        # fpn[2].shape == torch.Size([32, 64, 20, 20])

        # SSH
        feature1 = self.ssh1(fpn[0]) # feature1.shape == torch.Size([32, 64, 80, 80])
        feature2 = self.ssh2(fpn[1]) # feature2.shape == torch.Size([32, 64, 40, 40])
        feature3 = self.ssh3(fpn[2]) # feature3.shape == torch.Size([32, 64, 20, 20])
        features = [feature1, feature2, feature3]

        # # input [32, 64, 80, 80], output [32, 12800, 10]
        # test = self.LandmarkHead[0](feature1) # 12800 == 80*80(pixel)*2(anchor). Without back prop, the 12800 does not know which pixel they are mapped to.
        # print(test.shape) # 64 channels ->(conv)-> 20 channels -> 80*80*2 anchors predict 10 values for landmark
        # exit()

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)    # torch.Size([32, 16800, 4]) # 12800+3200+800
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)     # torch.Size([32, 16800, 2]) # 80*80*2+40*40*2+20*20*2
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1) # torch.Size([32, 16800, 10]) # conv1x1, set the outchannel to 10
        gaze_regressions = torch.cat([self.GazeHead[i](feature) for i, feature in enumerate(features)], dim=1) # torch.Size([32, 16800, 2]) # 16800 just values, do not know which pixel they are mapped to without back prop

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions, gaze_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions, gaze_regressions)
        return output