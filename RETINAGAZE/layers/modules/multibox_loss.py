import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
from data import cfg_mnet
GPU = cfg_mnet['gpu_train']

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
    
    # loss_l, loss_c, loss_landm = criterion(out, priors, targets)

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes # 2
        self.threshold = overlap_thresh # 0.35
        self.background_label = bkg_label # 0
        self.encode_target = encode_target # False
        self.use_prior_for_matching = prior_for_matching # True
        self.do_neg_mining = neg_mining # True
        self.negpos_ratio = neg_pos # 7
        self.neg_overlap = neg_overlap # 0.35
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # predictions.shape == ([32, 16800, 4]), ([32, 16800, 2]), ([32, 16800, 10], ([32, 16800, 2])
        # priors.shape == [16800, 4]
        # len(targets) == 32, targets[0].shape == [num_of_faces_in_this_image, 18]
        # 18 = 4(bbox) + 5*2(lmk) + 1(have_lmk) + 2(gaze) + 1(have_gaze)

        loc_data, conf_data, landm_data, gaze_data = predictions
        priors = priors
        num = loc_data.size(0) # 32 batch
        num_priors = (priors.size(0)) # 16800 anchor

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        gaze_t = torch.Tensor(num, num_priors, 2)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :4].data # bbox
            labels = targets[idx][:, -4].data # have_landmark, <- not true, -1 no_ldmk, 0 bkgd, 1 have_ldmk
            landms = targets[idx][:, 4:14].data # everyone's lmk
            gaze_labels = targets[idx][:, -1].data # have_landmark
            gazes  = targets[idx][:, -3:-1].data # everyone's lmk
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, landms, gaze_labels, gazes, loc_t, conf_t, landm_t, gaze_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()
            gaze_t = gaze_t.cuda()
        # print(loc_t.shape) # [32, 16800, 4]
        # print(conf_t.shape) # [32, 16800]
        # print(landm_t.shape) # [32, 16800, 10]
        # print(gaze_t.shape) # [32, 16800, 2]
        # exit()




        # LANDMARK LOSS

        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        # print(landm_p.shape) # [xxxx,10]
        landm_t = landm_t[pos_idx1].view(-1, 10)
        # print(landm_t.shape) # [xxxx,10]
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
        # print(loss_landm) # xxxxx.xxxx

        zeros = torch.tensor(0).cuda()
        ones = torch.tensor(1).cuda()hai

        # GAZE LOSS

        # gaze Loss (Smooth L1)
        # Shape: [batch,num_priors,2]
        pos2 = conf_t > zeros
        num_pos_gaze = pos2.long().sum(1, keepdim=True)
        N2 = max(num_pos_gaze.data.sum().float(), 1)
        pos_idx2 = pos2.unsqueeze(pos2.dim()).expand_as(gaze_data)
        gaze_p = gaze_data[pos_idx2].view(-1, 2)
        # print(gaze_p.shape) # [xxxx,2]
        gaze_t = gaze_t[pos_idx2].view(-1, 2)
        # print(gaze_t.shape) # [xxxx,2]
        loss_gaze = F.smooth_l1_loss(gaze_p, gaze_t, reduction='sum')
        # print(loss_gaze) # xxxxx.xxxx

        

        # LOCALIZATION LOSS

        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')





        # CLASSIFICATION LOSS

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1
        loss_gaze /= N2

        return loss_l, loss_c, loss_landm, loss_gaze
