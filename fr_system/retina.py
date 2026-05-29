import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from fr_system.retinaface.models.retinaface import RetinaFace
from fr_system.retinaface.data import cfg_mnet, cfg_re50
from fr_system.retinaface.layers.functions.prior_box import PriorBox
from fr_system.retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from utils.image_utils import float_to_uint8
from fr_system.retinaface.utils.box_utils import decode_batch, decode_landm_batch


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

class retinafacedetector(nn.Module):
    def __init__(self, model_path: str, network: str = 'resnet50', image_size: list[int] = [256, 256], confidence_threshold: float = 0.02, device: str = 'cuda'):
        super(retinafacedetector, self).__init__()
        self.cfg = cfg_mnet if network == 'mobile0.25' else cfg_re50
        print(self.cfg)
        self.device = torch.device(device)
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.net = self.load_model(self.net, model_path)
        self.priorbox = PriorBox(self.cfg, image_size=image_size)
        self.priors = self.priorbox.forward()

        self.net.eval()
        self.net.to(self.device)
        self.priors = self.priors.to(self.device)
        self.priors_data = self.priors.data

        self.confidence_threshold = confidence_threshold
        self.top_k = 5000
        self.nms_threshold = 0.4
        self.keep_top_k = 750
        self.normalization_vector = torch.tensor([104.0, 117.0, 123.0]).view(1,3,1,1).to(self.device)

    def load_model(self, model, pretrained_path):
        print('Loading pretrained model from {}'.format(pretrained_path))
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        check_keys(self.net, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def remove_prefix(self, state_dict, prefix):
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}


    def forward(self, img: torch.Tensor):
        #Expect img to be a tensor of shape (N, 3, H, W) in RGB format and uint8
        if img.dtype != torch.float32:
            raise ValueError("Input image tensor must be of type float32.")
        img = float_to_uint8(img)
        _, _, h, w = img.shape
        scale = torch.Tensor([w, h, w, h]).to(self.device)
        # Retina expect BGR format
        img = img[:, [2, 1, 0], :, :].float()
        # Subtract mean per channel
        img = img - self.normalization_vector
        loc, conf, landms = self.net(img)

        # boxes = decode(loc[0].data.squeeze(0), self.priors_data, self.cfg['variance'])

        boxes = decode_batch(loc.data, self.priors_data, self.cfg['variance'])

        # print(torch.allclose(boxes, boxes2[0]))

        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        # landms1 = decode_landm(landms[0].data.squeeze(0), self.priors_data, self.cfg['variance'])
        # Post-process the outputs (e.g., apply NMS, filter low-confidence detections)
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        landms = decode_landm_batch(landms.data, self.priors_data, self.cfg['variance'])
        # print(torch.allclose(landms1, landms0[0]))


        scale1 = scale1.to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()


        scores = conf.data.cpu().numpy()[:, :, 1]
        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)
        # print(inds)

        new_boxes =  np.zeros((boxes.shape[0], self.top_k, 4))
        new_landms = np.zeros((landms.shape[0], self.top_k, 10))
        new_scores = np.zeros((scores.shape[0], self.top_k))

        for i in range(boxes.shape[0]):
            i_inds = np.where(scores[i]>self.confidence_threshold)[0]
            n = len(i_inds)
            if n > self.top_k:
                n = self.top_k
            new_boxes[i, :n, :] = boxes[i, i_inds[:n], :]
            new_landms[i, :n, :] = landms[i, i_inds[:n], :]
            new_scores[i, :n] = scores[i, i_inds[:n]]

        
        boxes = new_boxes
        landms = new_landms
        scores = new_scores

        dets = np.zeros((boxes.shape[0], self.keep_top_k, 4 + 1))
        new_landms = np.zeros((landms.shape[0], self.keep_top_k, 10))
        new_scores = np.zeros((scores.shape[0], self.keep_top_k))
        for i in range(boxes.shape[0]):
            # keep top-K before NMS
            order = scores[i].argsort()[::-1][:self.top_k]
            boxes[i] = boxes[i][order]
            landms[i] = landms[i][order]
            scores[i] = scores[i][order]

            # do NMS
            det = np.hstack((boxes[i], scores[i, :, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(det, self.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets[i][:min(self.keep_top_k, len(keep))] = det[keep, :][:self.keep_top_k, :]
            new_landms[i][:min(self.keep_top_k, len(keep))] = landms[i][keep][:self.keep_top_k, :]
            new_scores[i][:min(self.keep_top_k, len(keep))] = scores[i][keep][:self.keep_top_k]

        # dets = np.concatenate((dets, new_landms, new_scores), axis=2)
        # new_landms = np.hstack((new_landms[:, :, :2],
        #                         new_landms[:, :, 2:4],
        new_landms = new_landms.reshape(new_landms.shape[0], new_landms.shape[1], 5, 2)
        dets = dets[:, :,:4]
        return dets, new_landms, new_scores

        # # show image
        # if args.save_image:
        #     for b in dets:
        #         if b[4] < args.vis_thres:
        #             continue
        #         text = "{:.4f}".format(b[4])
        #         b = list(map(int, b))
        #         cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        #         cx = b[0]
        #         cy = b[1] + 12
        #         cv2.putText(img_raw, text, (cx, cy),
        #                     cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        #         # landms
        #         cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        #         cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        #         cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        #         cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        #         cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        #     # save image

        #     name = "test.jpg"
        #     cv2.imwrite(name, img_raw)
