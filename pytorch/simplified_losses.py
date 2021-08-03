import torch
from torch.tensor import Tensor
import torch.nn.functional as F

from fvvdp_display_model import fvvdp_display_photo_gog
from hdrvdp_lpyr_dec import hdrvdp_lpyr_dec


def tf_log(X):    
    # The parameters were fitted to approximate PU21
    a = 0.456520040846940
    b = 1.070672820603428       
    return torch.log2(a*X+b)
   

class log_loss:
    def __init__(self):
        self.display_photo = fvvdp_display_photo_gog(Y_peak=200, gamma=2.2)

    def __call__(self, test_img: Tensor, ref_img: Tensor) -> Tensor:
        C, H, W = test_img.shape
        with torch.autograd.detect_anomaly():
            # Convert to linear absolute luminance
            test_img_Y = self.display_photo.forward(test_img.view(1, C, H, W))
            ref_img_Y = self.display_photo.forward(ref_img.view(1, C, H, W))

            return F.mse_loss( tf_log(test_img_Y), tf_log(ref_img_Y))


class ms_log_loss:
    def __init__(self):
        self.display_photo = fvvdp_display_photo_gog(Y_peak=200, gamma=2.2)

    def __call__(self, test_img: Tensor, ref_img: Tensor) -> Tensor:
        C, H, W = test_img.shape
        pix_per_deg = 50.
        lpyr = hdrvdp_lpyr_dec(W, H, pix_per_deg, test_img.device)

        with torch.autograd.detect_anomaly():
            # Convert to linear absolute luminance
            test_img_l = tf_log(self.display_photo.forward(test_img.view(1, C, H, W)))
            ref_img_l = tf_log(self.display_photo.forward(ref_img.view(1, C, H, W)))

            R = torch.cat( (test_img_l.view(1,C,H,W), ref_img_l.view(1,C,H,W)), dim=0)
            B_bands, B_gbands = lpyr.decompose( R )

            loss = 0
            for kk in range(len(B_bands)):
                loss += F.mse_loss(B_bands[kk][0,:,:,:], B_bands[kk][1,:,:,:])
                                
            return  loss


