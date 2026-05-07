from enum import Enum

class PatchifyStyle(Enum):
    VIT_STYLE = 0
    CONV1D_STYLE = 1

class UPSampleStyle(Enum):
    PIXEL_SHUFFLE = 0
    TRANSPOSE_CONV = 1
    BILINEAR=2
