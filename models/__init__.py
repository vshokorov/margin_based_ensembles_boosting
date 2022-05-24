from .vgg import *
from .wide_resnet import *
from .small_resnet import *
from .vision_transformer import *
import torchvision

class ResNet34:
    """
    Full args:
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    """
    def __init__(self):
        self.base = torchvision.models.resnet34
        self.kwargs = {}

class ResNet18:
    """
    Full args:
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    """
    def __init__(self):
        self.base = torchvision.models.resnet18
        self.kwargs = {}

class MobileNet:
    def __init__(self):
        """
        Full args:
            num_classes: int = 1000,
            width_mult: float = 1.0,
            inverted_residual_setting: Optional[List[List[int]]] = None,
            round_nearest: int = 8,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.2,
        """
        self.base = torchvision.models.mobilenet_v2
        self.kwargs = {}

class ViT:
    def __init__(self):
        """Vision Transformer as per https://arxiv.org/abs/2010.11929.
        
        Full args:
            image_size: int,
            patch_size: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_classes: int = 1000,
            representation_size: Optional[int] = None,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            conv_stem_configs: Optional[List[ConvStemConfig]] = None,
        """
        self.base  = torchvision.models.vit_b_16
        self.kwargs = {}
