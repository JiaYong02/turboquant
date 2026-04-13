"""TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate."""

__version__ = "0.1.0"

from .codebook import LloydMaxCodebook
from .qjl import QJL
from .quantizer_mse import TurboQuantMSE
from .quantizer_prod import TurboQuantProd
from .rotation import RandomRotation

__all__ = [
    "LloydMaxCodebook",
    "QJL",
    "RandomRotation",
    "TurboQuantMSE",
    "TurboQuantProd",
]

try:
    from .codebook_torch import LloydMaxCodebookTorch
    from .qjl_torch import QJLTorch
    from .quantizer_mse_torch import TurboQuantMSETorch
    from .quantizer_prod_torch import TurboQuantProdTorch
    from .rotation_torch import RandomRotationTorch

    __all__ += [
        "LloydMaxCodebookTorch",
        "QJLTorch",
        "RandomRotationTorch",
        "TurboQuantMSETorch",
        "TurboQuantProdTorch",
    ]
except ImportError:
    pass

try:
    from .integrations.hf_cache import TurboQuantCache

    __all__ += ["TurboQuantCache"]
except ImportError:
    pass
