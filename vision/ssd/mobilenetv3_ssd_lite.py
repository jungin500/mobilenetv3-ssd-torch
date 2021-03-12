import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from ..nn.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small, Block, hswish
from ..nn.mobilenetv3_custom import MobileNetV3 as CustomMobileNetV3

from .ssd import SSD
from .predictor import Predictor
from .config import mobilenetv1_ssd_config as config


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def create_mobilenetv3_large_ssd_lite(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False):
    base_net = MobileNetV3_Large().features

    source_layer_indexes = [ 15, 21 ]
    extras = ModuleList([
        Block(3, 960, 256, 512, hswish(), None, stride=2),
        Block(3, 512, 128, 256, hswish(), None, stride=2),
        Block(3, 256, 128, 256, hswish(), None, stride=2),
        Block(3, 256, 64, 64, hswish(), None, stride=2)
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=round(112 * width_mult), out_channels=6 * 4,
                        kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=960, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=round(112 * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=960, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_mobilenetv3_small_ssd_lite(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False):
    base_net = MobileNetV3_Small().features

    source_layer_indexes = [ 11, 17 ]
    extras = ModuleList([
        Block(3, 576, 256, 512, hswish(), None, stride=2),
        Block(3, 512, 128, 256, hswish(), None, stride=2),
        Block(3, 256, 128, 256, hswish(), None, stride=2),
        Block(3, 256, 64, 64, hswish(), None, stride=2)
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=round(48 * width_mult), out_channels=6 * 4,
                        kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=576, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=round(48 * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=576, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_custom_mobilenetv3_small_ssd_lite(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False):
    base_model = CustomMobileNetV3()
    # classifier = torch.nn.Sequential(
    #     torch.nn.AdaptiveAvgPool2d(output_size=1),
    #     torch.nn.Flatten(start_dim=1),
    #     torch.nn.Linear(576, 1024),
    #     torch.nn.Dropout(p=0.2),  # paper=0.8 but acc only achieves 10%
    #     torch.nn.Linear(1024, 1000)
    # )
    # model = torch.nn.Sequential(
    #     base_net,
    #     classifier
    # )
    # 
    # def xavier_init(m):
    #     if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             torch.nn.init.zeros_(m.bias)
    # 
    # model.apply(xavier_init)
    # base_model = torch.nn.DataParallel(base_model)  # Addes "module." prefix for state dicts
    
    # postprocessed_state_dict = torch.load('/shared/unmanaged-projects/ipcvl2021-mobilenetv3-torch/.checkpoints/' +
    #                                       'fe4f-mbv3-voc-experiment-after200epochs-continue-w1.00-r224-epoch0015-loss0.684-nextlr0.000328-acc0.785100.pth')
    postprocessed_state_dict = torch.load(r"C:\Workspace\study-projects\ipcvl2021-mobilenetv3-torch\.checkpoints" +
                                          r'\fe4f-mbv3-voc-experiment-after200epochs-continue-w1.00-r224-epoch0015-loss0.684-nextlr0.000328-acc0.785100.pth')
    base_model.load_state_dict(postprocessed_state_dict)
    base_net = base_model.features
    

    source_layer_indexes = [ 11, 17 ]
    extras = ModuleList([
        Block(3, 576, 256, 512, hswish(), None, stride=2),
        Block(3, 512, 128, 256, hswish(), None, stride=2),
        Block(3, 256, 128, 256, hswish(), None, stride=2),
        Block(3, 256, 64, 64, hswish(), None, stride=2)
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=round(48 * width_mult), out_channels=6 * 4,
                        kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=576, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=round(48 * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=576, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_mobilenetv3_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
