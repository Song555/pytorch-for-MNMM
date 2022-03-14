import matplotlib

matplotlib.use('Agg')
import sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
from skimage import io, img_as_float32
from MNMM.modules_s.generator import OcclusionAwareGenerator
from MNMM.modules_s.generator_s import OcclusionAwareGenerator_s
from modules_s.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f)

    generator_s = OcclusionAwareGenerator_s(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
    if not cpu:
        generator_s.cuda()


    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator_s.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator_s = DataParallelWithCallback(generator_s)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator_s.eval()
    kp_detector.eval()


    return generator_s, kp_detector


def make_animation(source_image, driving_video,  generator_s, kp_detector, relative=True,
                   adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator_s(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='config\MMI1.yaml', help="path to config")
    parser.add_argument("--checkpoint", default=r'checkpoints/00000099-checkpoint.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default=r'data/FY/12_aligned/frame_det_00_000001.png', help="path to source image")
    parser.add_argument("--driving_video", default=r'data/FY/12_aligned/frame_det_00_000007.png', help="path to driving video")
    parser.add_argument("--result_video", default=r'data/FY/12/7...png', help="path to output")
    # parser.add_argument("--source_image", default='data/0005_001_000.png', help="path to source image")
    # parser.add_argument("--driving_video", default='data/0005_001_0023.png', help="path to driving video")
    # parser.add_argument("--result_video", default='data/0005_001_0024.png', help="path to output")
    parser.add_argument("--relative", dest="relative", action="store_true",
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")
    # 基于关键点凸包的自适应运动尺度

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")
    # 从与源最关联的帧生成。（仅适用于面，需要面对齐库）

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,
                        help="Set frame to start from.")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    #source_image = img_as_float32(io.imread(opt.source_image))
    source_image = imageio.imread(opt.source_image)  # 读取source image
    reader = imageio.imread(opt.driving_video)  # 读取source image
    #reader = imageio.get_reader(opt.driving_video)  # 读取driveing video
    #reader = img_as_float32(io.imread(opt.driving_video))
    #fps = reader.get_meta_data()['fps']
    #driving_video = []
    #try:
    #    for im in reader:
    #        driving_video.append(im)
    #except RuntimeError:
    #    pass
    #reader.close()
    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = resize(reader, (256, 256))[..., :3]
    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    driving_frame = torch.tensor(driving_video[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    #driving_b = np.array(driving_video, dtype='float32')
    #source = source.transpose((2, 0, 1))
    #driving_frame = driving_b.transpose((2, 0, 1))
    #driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]  # 大小
    generator_s, kp_detector= load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint,
                                                           cpu=opt.cpu)
    # 加载预训练模型中的生成器和关键点检测器
    #source = source.cuda()
    print(source.shape)
    kp_source = kp_detector(source)
    #driving_frame = driving_b.cuda()
    kp_driving = kp_detector(driving_frame)

    out, middleout = generator_s(source, kp_source=kp_source, kp_driving=kp_driving)
    final = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
    print(final.shape)

    imageio.imwrite(opt.result_video, final)  # 生成动画
    # img_as_ubyte将图像转换为8位无符号整数格式

