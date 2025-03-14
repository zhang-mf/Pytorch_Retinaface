import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes'] # 'min_sizes': [[16, 32], [64, 128], [256, 512]]
        self.steps = cfg['steps'] # 'steps': [8, 16, 32]
        self.clip = cfg['clip'] # False
        self.image_size = image_size # (640,640)
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps] 
        # 640/8,/16,/32 [[80, 80], [40, 40], [20, 20]]
        self.name = "s"

    def forward(self):
        # zmf
        # anchors distribution: For every pixel in downsampled f-maps (3 different-scale f-maps), map back to orig-size image
        # num of anchor centers: 80*80 + 40*40 + 20*20 = 8400 (cx,cy)
        # anchor size: For every downsampled f-map, assign 2 anchor sizes to each pixel (s_kx,s_ky)
        # num of anchors: 8400*2 = 16800
        # anchor shape: square
        #
        # len(anchors) == 67200 == 16800 * 4
        # first anchor : [0.00625,0.00625,0.025,0.025]

        anchors = []
        for k, f in enumerate(self.feature_maps): # [80,80], [40,40], [20,20]
            # 12800 + 3200 + 800
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                # 80*80*2 + 40*40*2 + 20*20*2
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky] 

        

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
