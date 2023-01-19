import torch
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import SingleStageDetector


@DETECTORS.register_module('YOLOV3', force=True)
class YOLOV3(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(YOLOV3, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained, init_cfg)

    def forward(self, img, img_metas, flag=False, return_loss=True, **kwargs):
        if flag:
            return self.forward_dummy(img)
        else:
            if torch.onnx.is_in_onnx_export():
                assert len(img_metas) == 1
                return self.onnx_export(img[0], img_metas[0])

            if return_loss:
                return self.forward_train(img, img_metas, **kwargs)
            else:
                return self.forward_test(img, img_metas, **kwargs)

    def onnx_export(self, img, img_metas):
        x = self.extract_feat(img)
        outs = self.bbox_head.forward(x)
        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape

        det_bboxes, det_labels = self.bbox_head.onnx_export(*outs, img_metas)

        return det_bboxes, det_labels