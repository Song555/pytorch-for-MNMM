from tqdm import trange
import torch

from torch.utils.data import DataLoader

from logger import Logger
from MNMM.modules_s.model import GeneratorFullModel, DiscriminatorFullModel

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater


def train(config,generator,generator_s, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer_generator_s = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, generator_s.parameters()), 'initial_lr': 2.0e-4}],lr=train_params['lr_generator_s'], betas=(0.5, 0.999))
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    #优化器

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator_s, generator, discriminator, kp_detector,
                                      optimizer_generator_s, optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
    else:
        start_epoch = 0


    #学习率
    #scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
    #                                  last_epoch=start_epoch - 1)
    scheduler_generator_s = MultiStepLR(optimizer_generator_s, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])#数据集ab
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)
    #数据加载



    #教师模型
    generator_full = GeneratorFullModel(kp_detector, generator,generator_s, discriminator, train_params)#生成器
    discriminator_full = DiscriminatorFullModel(kp_detector, generator_s, discriminator, train_params)#鉴别器

    if torch.cuda.is_available():#转到GPU
        generator_full_s = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full_s = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):#开始训练
            for x in dataloader:#遍历数据集
                losses_generator, generated_s,generated_c = generator_full_s(x)#生成器输出生成图片和损失
                #此时损失包括
                generated_s['Magnified'] = generated_c['prediction']
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)#损失
                loss.backward()#后向传播
                optimizer_generator_s.step()#生成器优化器梯度下降
                optimizer_generator_s.zero_grad()#生成器优化器梯度清零
                #optimizer_kp_detector.step()#关键点检测器优化器梯度下降
                #optimizer_kp_detector.zero_grad()#关键点检测器优化器梯度清零

                if train_params['loss_weights']['generator_gan'] != 0:
                    #optimizer_discriminator.zero_grad()#鉴别器优化器梯度清零
                    losses_discriminator_s = discriminator_full_s(x, generated_s)#返回GAN网络的损失
                    loss_values = [val.mean() for val in losses_discriminator_s.values()]#求loss均值
                    loss = sum(loss_values)

                    loss.backward()#后向传播
                    #optimizer_discriminator.step()#梯度下降
                    #optimizer_discriminator.zero_grad()#梯度清零
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler_generator_s.step()
            #scheduler_discriminator.step()
            #scheduler_kp_detector.step()#学习率梯度下降
            
            logger.log_epoch(epoch, {'generator': generator_s,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator_s,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector}, inp=x, out=generated_s)
