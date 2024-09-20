import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import caffe_pb2
from utils import conv2DBatchNormRelu, residualBlockPSP, pyramidPooling

pspnet_specs = {
    "pascal": {"n_classes": 21, "input_size": (473, 473), "block_config": [3, 4, 23, 3]},
    "cityscapes": {"n_classes": 19, "input_size": (713, 713), "block_config": [3, 4, 23, 3]},
    "ade20k": {"n_classes": 150, "input_size": (473, 473), "block_config": [3, 4, 6, 3]},
}


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            target.device
        )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


class PSPnet(nn.Module):

    """
    Pyramid Scene Parsing Network
    URL: https://arxiv.org/abs/1612.01105
    References:
    1) Original Author's code: https://github.com/hszhao/PSPNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/PSPNet-tensorflow
    Visualization:
    http://dgschwend.github.io/netscope/#/gist/6bfb59e6a3cfcb4e2bb8d47f827c2928
    """

    def __init__(
        self, n_classes=7, block_config=[3, 4, 6, 3], input_size=(800, 600), version=None
    ):

        super(PSPnet, self).__init__()

        self.block_config = (
            pspnet_specs[version]["block_config"] if version is not None else block_config
        )
        self.n_classes = pspnet_specs[version]["n_classes"] if version is not None else n_classes
        self.input_size = pspnet_specs[version]["input_size"] if version is not None else input_size

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(
            in_channels=3, k_size=3, n_filters=64, padding=1, stride=2, bias=False
        )
        self.convbnrelu1_2 = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=64, padding=1, stride=1, bias=False
        )
        self.convbnrelu1_3 = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=128, padding=1, stride=1, bias=False
        )

        # Vanilla Residual Blocks
        self.res_block2 = residualBlockPSP(self.block_config[0], 128, 64, 256, 1, 1)
        self.res_block3 = residualBlockPSP(self.block_config[1], 256, 128, 512, 2, 1)

        # Dilated Residual Blocks
        self.res_block4 = residualBlockPSP(self.block_config[2], 512, 256, 1024, 1, 2)
        self.res_block5 = residualBlockPSP(self.block_config[3], 1024, 512, 2048, 1, 4)

        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(2048, [6, 3, 2, 1])

        # Final conv layers
        self.cbr_final = conv2DBatchNormRelu(4096, 512, 3, 1, 1, False)
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        self.classification = nn.Conv2d(512, self.n_classes, 1, 1, 0)

        # Auxiliary layers for training
        self.convbnrelu4_aux = conv2DBatchNormRelu(
            in_channels=1024, k_size=3, n_filters=256, padding=1, stride=1, bias=False
        )
        self.aux_cls = nn.Conv2d(256, self.n_classes, 1, 1, 0)

        # Define auxiliary loss function
        self.loss = multi_scale_cross_entropy2d

        self.load_pretrained_model("pspnet50_ADE20K.caffemodel")

    def forward(self, x):
        inp_shape = x.shape[2:]

        # H, W -> H/2, W/2
        x = self.convbnrelu1_1(x)
        x = self.convbnrelu1_2(x)
        x = self.convbnrelu1_3(x)

        # H/2, W/2 -> H/4, W/4
        x = F.max_pool2d(x, 3, 2, 1)

        # H/4, W/4 -> H/8, W/8
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        # Auxiliary layers for training
        if self.training:
            x_aux = self.convbnrelu4_aux(x)
            x_aux = self.dropout(x_aux)
            x_aux = self.aux_cls(x_aux)

        x = self.res_block5(x)

        x = self.pyramid_pooling(x)

        x = self.cbr_final(x)
        x = self.dropout(x)

        x = self.classification(x)
        x = F.interpolate(x, size=inp_shape, mode="bilinear", align_corners=True)

        if self.training:
            # return (x, x_aux)
            return x
        else:  # eval mode
            return x

    def load_pretrained_model(self, model_path):
        """
        Load weights from caffemodel w/o caffe dependency
        and plug them in corresponding modules
        """
        # My eyes and my heart both hurt when writing this method

        # Only care about layer_types that have trainable parameters
        ltypes = ["BNData", "ConvolutionData", "HoleConvolutionData"]

        def _get_layer_params(layer, ltype):

            if ltype == "BNData":
                gamma = np.array(layer.blobs[0].data)
                beta = np.array(layer.blobs[1].data)
                mean = np.array(layer.blobs[2].data)
                var = np.array(layer.blobs[3].data)
                return [mean, var, gamma, beta]

            elif ltype in ["ConvolutionData", "HoleConvolutionData"]:
                is_bias = layer.convolution_param.bias_term
                weights = np.array(layer.blobs[0].data)
                bias = []
                if is_bias:
                    bias = np.array(layer.blobs[1].data)
                return [weights, bias]

            elif ltype == "InnerProduct":
                raise Exception("Fully connected layers {}, not supported".format(ltype))

            else:
                raise Exception("Unkown layer type {}".format(ltype))

        net = caffe_pb2.NetParameter()
        with open(model_path, "rb") as model_file:
            net.MergeFromString(model_file.read())

        # dict formatted as ->  key:<layer_name> :: value:<layer_type>
        layer_types = {}
        # dict formatted as ->  key:<layer_name> :: value:[<list_of_params>]
        layer_params = {}

        for l in net.layer:
            lname = l.name
            ltype = l.type
            if ltype in ltypes:
                print("Processing layer {}".format(lname))
                layer_types[lname] = ltype
                layer_params[lname] = _get_layer_params(l, ltype)

        # Set affine=False for all batchnorm modules
        def _no_affine_bn(module=None):
            if isinstance(module, nn.BatchNorm2d):
                module.affine = False

            if len([m for m in module.children()]) > 0:
                for child in module.children():
                    _no_affine_bn(child)

        # _no_affine_bn(self)

        def _transfer_conv(layer_name, module):
            weights, bias = layer_params[layer_name]
            w_shape = np.array(module.weight.size())

            print(
                "CONV {}: Original {} and trans weights {}".format(
                    layer_name, w_shape, weights.shape
                )
            )

            module.weight.data.copy_(torch.from_numpy(weights).view_as(module.weight))

            if len(bias) != 0:
                b_shape = np.array(module.bias.size())
                print(
                    "CONV {}: Original {} and trans bias {}".format(layer_name, b_shape, bias.shape)
                )
                module.bias.data.copy_(torch.from_numpy(bias).view_as(module.bias))

        def _transfer_conv_bn(conv_layer_name, mother_module):
            conv_module = mother_module[0]
            bn_module = mother_module[1]

            _transfer_conv(conv_layer_name, conv_module)

            mean, var, gamma, beta = layer_params[conv_layer_name + "/bn"]
            print(
                "BN {}: Original {} and trans weights {}".format(
                    conv_layer_name, bn_module.running_mean.size(), mean.shape
                )
            )
            bn_module.running_mean.copy_(torch.from_numpy(mean).view_as(bn_module.running_mean))
            bn_module.running_var.copy_(torch.from_numpy(var).view_as(bn_module.running_var))
            bn_module.weight.data.copy_(torch.from_numpy(gamma).view_as(bn_module.weight))
            bn_module.bias.data.copy_(torch.from_numpy(beta).view_as(bn_module.bias))

        def _transfer_residual(prefix, block):
            block_module, n_layers = block[0], block[1]

            bottleneck = block_module.layers[0]
            bottleneck_conv_bn_dic = {
                prefix + "_1_1x1_reduce": bottleneck.cbr1.cbr_unit,
                prefix + "_1_3x3": bottleneck.cbr2.cbr_unit,
                prefix + "_1_1x1_proj": bottleneck.cb4.cb_unit,
                prefix + "_1_1x1_increase": bottleneck.cb3.cb_unit,
            }

            for k, v in bottleneck_conv_bn_dic.items():
                _transfer_conv_bn(k, v)

            for layer_idx in range(2, n_layers + 1):
                residual_layer = block_module.layers[layer_idx - 1]
                residual_conv_bn_dic = {
                    "_".join(
                        map(str, [prefix, layer_idx, "1x1_reduce"])
                    ): residual_layer.cbr1.cbr_unit,
                    "_".join(map(str, [prefix, layer_idx, "3x3"])): residual_layer.cbr2.cbr_unit,
                    "_".join(
                        map(str, [prefix, layer_idx, "1x1_increase"])
                    ): residual_layer.cb3.cb_unit,
                }

                for k, v in residual_conv_bn_dic.items():
                    _transfer_conv_bn(k, v)

        convbn_layer_mapping = {
            "conv1_1_3x3_s2": self.convbnrelu1_1.cbr_unit,
            "conv1_2_3x3": self.convbnrelu1_2.cbr_unit,
            "conv1_3_3x3": self.convbnrelu1_3.cbr_unit,
            "conv5_3_pool6_conv": self.pyramid_pooling.paths[0].cbr_unit,
            "conv5_3_pool3_conv": self.pyramid_pooling.paths[1].cbr_unit,
            "conv5_3_pool2_conv": self.pyramid_pooling.paths[2].cbr_unit,
            "conv5_3_pool1_conv": self.pyramid_pooling.paths[3].cbr_unit,
            "conv5_4": self.cbr_final.cbr_unit,
            "conv4_" + str(self.block_config[2] + 1): self.convbnrelu4_aux.cbr_unit,
        }  # Auxiliary layers for training

        residual_layers = {
            "conv2": [self.res_block2, self.block_config[0]],
            "conv3": [self.res_block3, self.block_config[1]],
            "conv4": [self.res_block4, self.block_config[2]],
            "conv5": [self.res_block5, self.block_config[3]],
        }

        # Transfer weights for all non-residual conv+bn layers
        for k, v in convbn_layer_mapping.items():
            _transfer_conv_bn(k, v)

        # Transfer weights for final non-bn conv layer
        # _transfer_conv("conv6", self.classification)
        # _transfer_conv("conv6_1", self.aux_cls)

        # Transfer weights for all residual layers
        for k, v in residual_layers.items():
            _transfer_residual(k, v)





