from ..builder import HEADS, build_head, build_roi_extractor
import torch
import clip
import torch.nn as nn
from .standard_roi_head import StandardRoIHead
import torch.nn.functional as F
from torch import distributed as dist
from .visualize import visualize_oam_boxes
from mmdet.core import (bbox2roi, bbox2result, build_assigner, merge_aug_bboxes, bbox_overlaps, 
                        build_sampler, multiclass_nms)
from mmdet.core.bbox.coder.delta_xywh_bbox_coder import delta2bbox
from mmcv.ops.roi_align import roi_align
import ipdb
from .class_name import *
from tqdm import tqdm
import os.path as osp
import time
from .roi_extractors.single_level_roi_extractor import SingleRoIExtractor

def weighted_iou_regression_loss(iou_pred, iou_target, weight, avg_factor=None):
    """

    :param iou_pred: tensor of shape (batch*A*width*height) or (batch*num_pos)
    :param iou_target: tensor of shape (batch*A*width*height)Or tensor of shape (batch*num_pos), store the iou between
          predicted boxes and its corresponding groundtruth boxes for the positives and the iou between the predicted
          boxes and anchors for negatives.
    :param weight: tensor of shape (batch*A*width*height) or (batch*num_pos), 1 for positives and 0 for negatives and neutrals.
    :param avg_factor:
    :return:
    """
    # iou_pred_sigmoid = iou_pred.sigmoid()
    # iou_target = iou_target.detach()

    # L2 loss.
    # loss = torch.pow((iou_pred_sigmoid - iou_target), 2)*weight
    # ipdb.set_trace()

    # Binary cross-entropy loss for the positive examples
    loss = F.binary_cross_entropy_with_logits(iou_pred, iou_target, reduction='none')* weight

    return torch.sum(loss)[None] / avg_factor
    
@HEADS.register_module()
class StandardRoIHeadTEXT(StandardRoIHead):
    """RoI head for Double Head RCNN

    https://arxiv.org/abs/1904.06493
    """

    def __init__(self, bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 load_feature=True,
                 prompt_path=None, **kwargs):
        super(StandardRoIHeadTEXT, self).__init__(bbox_roi_extractor=bbox_roi_extractor,
                                              bbox_head=bbox_head,
                                              mask_roi_extractor=mask_roi_extractor,
                                              mask_head=mask_head,
                                              shared_head=shared_head,
                                              train_cfg=train_cfg,
                                              test_cfg=test_cfg,
                                              **kwargs)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if bbox_head.num_classes == 80:
            self.CLASSES = COCO_CLASSES
            dataset = 'coco'
        elif bbox_head.num_classes == 20:
            self.CLASSES = VOC_CLASSES
            dataset = 'voc'
        else:
            self.CLASSES = LVIS_CLASSES
            dataset = 'lvis'
        self.num_classes = len(self.CLASSES)
        print('num_classes:',self.num_classes)
        
        if self.num_classes == 1203:
            self.base_label_ids = lvis_base_label_ids
            self.novel_label_ids = torch.tensor(lvis_novel_label_ids, device=device)
            # self.novel_index = F.pad(torch.bincount(self.novel_label_ids),(0,self.num_classes-self.novel_label_ids.max())).bool()
            # 没用到
        elif self.num_classes == 20:
            self.novel_label_ids = torch.tensor(voc_novel_label_ids, device=device)
            # self.novel_index = F.pad(torch.bincount(self.novel_label_ids),(0,self.num_classes-self.novel_label_ids.max())).bool()
        elif self.num_classes == 80:
            self.base_label_ids = coco_base_label_ids
            self.novel_label_ids = torch.tensor(coco_novel_label_ids, device=device)
            # self.novel_index = F.pad(torch.bincount(self.novel_label_ids),(0,self.num_classes-self.novel_label_ids.max())).bool()
        self.clip_model, self.preprocess = clip.load('RN50', device)
        self.clip_model.eval().float()
        # self.reporter = MemReporter(self.clip_model)
        for param in self.clip_model.parameters():
            param.requires_grad = False
   
        self.text_features_for_classes = []
        self.iters = 0
        self.ensemble = False
        self.load_feature = load_feature
        print('ensemble:{}'.format(self.ensemble))
        print('prompt path',prompt_path)
        save_path = 'epoch_10_embedding.pth'
        if prompt_path is not None:
            save_path = prompt_path
        # save_path = dataset + '_text_embedding.pt'
        time_start = time.time()
        if osp.exists(save_path):
            self.text_features_for_classes = torch.load(save_path).to(device).squeeze()
        else:
            self.clip_model, self.preprocess = clip.load('RN50', device)
            self.clip_model.eval().float()

            for param in self.clip_model.parameters():
                param.requires_grad = False
            for template in tqdm(template_list):
                text_features_for_classes = torch.cat([self.clip_model.encode_text(clip.tokenize(template.format(c)).to(device)).detach() for c in self.CLASSES])
                self.text_features_for_classes.append(F.normalize(text_features_for_classes,dim=-1))

            # ipdb.set_trace()
            self.text_features_for_classes = torch.stack(self.text_features_for_classes).mean(dim=0)
            torch.save(self.text_features_for_classes.detach().cpu(),save_path)
        self.text_features_for_classes = self.text_features_for_classes.float()
        self.text_features_for_classes = F.normalize(self.text_features_for_classes,dim=-1)
        # ipdb.set_trace()
        # reporter.report()
        print('text embedding finished, {} passed'.format(time.time()-time_start))
        self.bg_embedding = nn.Linear(1,1024)
        # self.projection = nn.Linear(1024,1024)
        
        self.roialign = SingleRoIExtractor(roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0), out_channels=2048, featmap_strides=[32])

        # self.temperature = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.temperature.data.fill_(0.01)
        self.temperature = 0.01
        
        # if self.ensemble:
        #     self.projection_for_image = nn.Linear(1024,512)
        #     nn.init.xavier_uniform_(self.projection_for_image.weight)
        #     nn.init.constant_(self.projection_for_image.bias, 0)

        nn.init.xavier_uniform_(self.bg_embedding.weight)
        nn.init.constant_(self.bg_embedding.bias, 0)

        # nn.init.xavier_uniform_(self.projection.weight)
        # nn.init.constant_(self.projection.bias, 0)
    
    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      top_level_feature,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            region_embeddings, VLM_embedding, text_features, labels = self._bbox_forward_train(x,top_level_feature,
                                                    sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            # losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        # if self.with_mask:
        #     mask_results = self._mask_forward_train(x, sampling_results,
        #                                             bbox_results['bbox_feats'],
        #                                             gt_masks, img_metas)
        #     losses.update(mask_results['loss_mask'])

        return region_embeddings, VLM_embedding, text_features, labels
    
    
    def clip_image_forward_align(self, img, bboxes):
        # Top_level_feature = self.Top_level_feature_extract(img)
        Top_level_feature = img
        # ipdb.set_trace()
        feature = []
        feature.append(Top_level_feature)
        cropped_embeddings = self.roialign(tuple(feature), bboxes)
        return cropped_embeddings

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        # rois = rois.float()
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        region_embeddings = self.bbox_head.forward_embedding(bbox_feats)
        # ipdb.set_trace()
        
        bbox_pred = self.bbox_head(region_embeddings)
        bbox_results = dict(
            bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results, region_embeddings

    def _bbox_forward_train(self, x, top_level_feature, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        num_total_pos = sum([max(res.pos_inds.numel(), 1) for res in sampling_results])
        input_one = x[0].new_ones(1)
        bg_class_embedding = self.bg_embedding(input_one).reshape(1, 1024)
        bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding, p=2, dim=1)
        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        # ipdb.set_trace()
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        labels, label_weights, bbox_target, bbox_weights = bbox_targets
        
        pred_bbox = delta2bbox(rois[:,1:], bbox_results['bbox_pred'])
        target_bbox = delta2bbox(rois[:,1:], bbox_target)
        
        bbox_weight_list = torch.split(bbox_weights, 1, -1)
        bbox_weight = bbox_weight_list[0]
        
        iou = torch.unsqueeze(bbox_overlaps(pred_bbox, target_bbox, is_aligned=True), dim=1) # (batch*width_i*height_i*A)
        region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)
        text_features = torch.cat([self.text_features_for_classes, bg_class_embedding], dim=0)
        
        cls_score_text = region_embeddings @ text_features.T
        cls_score_text = cls_score_text / self.temperature
        
        cropped_embeddings = self.clip_image_forward_align(top_level_feature, rois)
        VLM_embedding = self.clip_model.visual.attnpool(cropped_embeddings)
        
        #0.009#0.008#0.007
             
        # cls_score_text[:,self.novel_label_ids] = -1e11  # 貌似不需要用,用了损失函数非常大,因为把一些值变0了,也可以该labels上
        # ipdb.set_trace()
        # text_cls_loss = F.cross_entropy(cls_score_text / self.temperature, labels, reduction='mean')
        
        # loss_bbox = self.bbox_head.loss(cls_score_text,
        #     bbox_results['bbox_pred'], rois,
        #     *bbox_targets)

        # bbox_results.update(loss_bbox=loss_bbox)
        return region_embeddings, VLM_embedding, text_features, labels


    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.bool))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.bool))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results


    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    def simple_test_bboxes(self,
                           x,
                           top_level_feature,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        # get origin input shape to support onnx dynamic input shape
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        rois = bbox2roi(proposals)

        bbox_results,region_embeddings = self._bbox_forward(x,rois)
        
        region_embeddings = torch.nn.functional.normalize(region_embeddings,p=2,dim=1)
        input_one = x[0].new_ones(1)
        bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
        bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding,p=2,dim=1)
        text_features = torch.cat([self.text_features_for_classes,bg_class_embedding],dim=0)
        #-----------------------------------------------------
        # """
        # ipdb.set_trace()
        cls_score_text = region_embeddings @ text_features.T
        cls_score_text = cls_score_text / self.temperature
        #0.009#0.008#0.007
        #--------------------------------------------
        # """
        cropped_embeddings = self.clip_image_forward_align(top_level_feature, rois)
        VLM_embedding = self.clip_model.visual.attnpool(cropped_embeddings)
        
        cls_score_VLM = VLM_embedding @ text_features.T
        cls_score_VLM = cls_score_VLM / self.temperature
        #0.009#0.008#0.007
           
        a = 1/3

        # cls_score= torch.where(self.novel_index,cls_score_VLM**(1-a)*cls_score_text**a,
        #                 cls_score_text**(1-a)*cls_score_VLM**a)

        # cls_score_align= torch.where(self.novel_index,cls_score_clip_align**(1-a)*cls_score_text**a,
                        #    cls_score_text**(1-a)*cls_score_clip_align**a)
        cls_score = cls_score_text
        # cls_score = cls_score_image
        # ipdb.set_trace()
        
        bbox_pred = bbox_results['bbox_pred']
        # bbox_pred = torch.squeeze(bbox_results['bbox_pred'][iou_pred.sort(dim=0, descending = True)[1]]) 
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def simple_test(self,
                    x,
                    img,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, segm_results
            else:
                return det_bboxes, det_labels

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    