from torch import nn
import torch
import torch.nn.functional as F
from modules_s.util import AntiAliasInterpolation2d, make_coordinate_grid
from torchvision import models
import numpy as np
from torch.autograd import grad


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator,generator_s, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator #generator_S
        self.generator_s = generator_s
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def forward(self, x):
        kp_source = self.kp_extractor(x['source'])
        kp_driving_b = self.kp_extractor(x['driving_b'])
        kp_driving_c = self.kp_extractor(x['driving_c'])

        generated_c = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving_c) #source+kp_driving_C，返回densemotion
        #学生模型 source+kp_driving_B ,输出generated_s，返回densemotion（过了unet后的）
        generated_c.update({'kp_source': kp_source, 'kp_driving_c': kp_driving_c})
        # 下面的loss全是generated_s和真值的loss，教师模型不用训练，不要loss。再加一组loss，loss_values['middlelayer']
        generated_s, _ = self.generator_s(x['source'], kp_source=kp_source, kp_driving=kp_driving_b)
        generated_s.update({'kp_source': kp_source, 'kp_driving_b': kp_driving_b})



        loss_values = {}

        pyramide_real = self.pyramid(x['driving_c']) #x[driving_C]
        #pyramide_real = self.pyramid(generated_c['prediction']) #generated_C[prediction]
        pyramide_generated_s = self.pyramid(generated_s['prediction'])

        pyramide_realdeformation_c = (255 * generated_c['deformed'])  # generated_C[prediction]
        pyramide_generateddeformation_s = (255 * generated_s['deformed'])

        pyramide_realocclusion_c = (255 * generated_c['occlusion_map'])  # generated_C[prediction]
        pyramide_generateocclusion_s = (255 * generated_s['occlusion_map'])


        if sum(self.loss_weights['middlelayer_d']) != 0:
            #value_total = 0
            #for scale in self.scales:
            #    y_vgg = self.vgg(pyramide_realdeformation_c['deformation' + str(scale)])
            #    x_vgg = self.vgg(pyramide_generateddeformation_s['deformation' + str(scale)])
            #    # s_vgg = self.vgg(pyramide_generated_s['prediction_' + str(scale)])

            #     for i, weight in enumerate(self.loss_weights['middlelayer_d']):
            #        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
            #        value_total += self.loss_weights['middlelayer_d'][i] * value
            #    loss_values['middlelayer_d'] = value_total
            value = torch.abs(pyramide_realdeformation_c - pyramide_generateddeformation_s).mean() * 10
            loss_values['middlelayer_d'] = value

        if sum(self.loss_weights['middlelayer_oc']) != 0:
            #value_total = 0
            #for scale in self.scales:
            #    y_vgg = self.vgg(pyramide_realocclusion_c['deformation' + str(scale)])
            #    x_vgg = self.vgg(pyramide_generateocclusion_s['deformation' + str(scale)])
                # s_vgg = self.vgg(pyramide_generated_s['prediction_' + str(scale)])

            #    for i, weight in enumerate(self.loss_weights['middlelayer_oc']):
            #        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
            #        value_total += self.loss_weights['middlelayer_oc'][i] * value
            #    loss_values['middlelayer_oc'] = value_total
            value = torch.abs(pyramide_realocclusion_c - pyramide_generateocclusion_s).mean() * 10
            loss_values['middlelayer_oc'] = value

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated_s['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])
                #s_vgg = self.vgg(pyramide_generated_s['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:#需要思考discriminator_maps_generated的kp
            discriminator_maps_generated = self.discriminator(pyramide_generated_s, kp=detach_kp(kp_driving_c)) #kp_driving_Cs,所有的参数调整确认
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving_c))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving_c'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving_c'])
            transformed_kp = self.kp_extractor(transformed_frame)

            generated_s['transformed_frame'] = transformed_frame
            generated_s['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving_c['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                                    transformed_kp['jacobian'])

                normed_driving = torch.inverse(kp_driving_c['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        return loss_values, generated_s,generated_c


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        kp_driving = generated['kp_driving']
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values
