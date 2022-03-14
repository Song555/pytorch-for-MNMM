from modules_s.Unet import Unet
from modules_s.dense_motion import DenseMotionNetwork
import torch
from torch import nn
import torch.nn.functional as F

class Student(nn.Module):

    def __init__(self, num_channels, num_kp,estimate_occlusion_map=False,dense_motion_params=None):
        super(Student, self).__init__()
        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None


    def forward(self,source_image, kp_driving, kp_source):
        output_student = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_student['mask'] = dense_motion['mask']
            output_student['sparse_deformed'] = dense_motion['sparse_deformed']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']  # 1 64 64 2
            deformation = deformation.permute(0, 3, 1, 2)  # 1 2 64 64
            deformation = deformation.view(1, -1, 572, 572)
            _, H_d, W_d, _ = deformation.shape
            Unet1 = Unet(n_channels=H_d, n_classes=2)
            deformation = Unet1(deformation)
            deformation = deformation.view(1, 2, 64, 64)  #
            output_student['deformation'] = deformation


        if occlusion_map is not None:
            if occlusion_map.shape[2] != 64 or occlusion_map.shape[3] != 64:
                occlusion_map = F.interpolate(occlusion_map, size=(64,64), mode='bilinear')
                occlusion_map = occlusion_map.view(1, -1, 572, 572)
                _, H_o, W_o, _ = occlusion_map.shape
                Unet2 = Unet(n_channels=H_o, n_classes=2)
                occlusion_map = Unet2(occlusion_map)
                occlusion_map = occlusion_map.view(1, 1, 64, 64)
                output_student['occlusion_map'] = occlusion_map

        output_student.requires_grad = True

        return output_student










