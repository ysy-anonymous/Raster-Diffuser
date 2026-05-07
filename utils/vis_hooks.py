import torch
import torch.nn.functional as F


class CustomAttentionExtractor:
    def __init__(self, model, q_name, k_name, num_heads=1, q_is_projection=False):
        self.model = model
        self.modules = dict(model.named_modules())
        self.q_name = q_name
        self.k_name = k_name
        self.num_heads = num_heads
        self.q_is_projection = q_is_projection
        self.handles = []
        self.q = None
        self.k = None

    def register(self):
        def q_hook(mod, inp, out):
            self.q = out.detach()

        def k_hook(mod, inp, out):
            self.k = out.detach()

        self.handles.append(self.modules[self.q_name].register_forward_hook(q_hook))
        self.handles.append(self.modules[self.k_name].register_forward_hook(k_hook))

    def reset(self):
        self.q = None
        self.k = None

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _to_bld(self, x):
        # convert to [B, L, D]
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got shape={tuple(x.shape)}")

        # heuristic:
        # Conv1d-ish feature often [B, C, L]
        # token feature often [B, L, D]
        if x.shape[1] <= x.shape[2]:
            # likely [B, L, D]
            return x
        else:
            # likely [B, D, L] -> [B, L, D]
            return x.transpose(1, 2)

    def get_attention(self):
        q = self._to_bld(self.q)
        k = self._to_bld(self.k)

        B, Lq, Dq = q.shape
        Bk, Lk, Dk = k.shape
        assert B == Bk, "Batch mismatch"

        # if dims differ, you cannot multiply directly
        # then q_name is probably not the real query feature.
        if Dq != Dk:
            raise ValueError(
                f"Query dim ({Dq}) != key dim ({Dk}). "
                f"Your q_name may not match the attention input dimension."
            )

        H = self.num_heads
        assert Dq % H == 0, "Embedding dim must be divisible by num_heads"
        Dh = Dq // H

        q = q.view(B, Lq, H, Dh).transpose(1, 2)   # [B, H, Lq, Dh]
        k = k.view(B, Lk, H, Dh).transpose(1, 2)   # [B, H, Lk, Dh]

        attn = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)
        attn = torch.softmax(attn, dim=-1)         # [B, H, Lq, Lk]
        return attn.detach().cpu().numpy()
    

class FiLMParamExtractor:
    def __init__(self, model, film_names):
        self.model = model
        self.modules = dict(model.named_modules())
        self.film_names = film_names
        self.handles = []
        self.raw = {}

    def register(self):
        for name in self.film_names:
            def make_hook(n):
                def hook(mod, inp, out):
                    self.raw[n] = out.detach()
                return hook
            self.handles.append(self.modules[name].register_forward_hook(make_hook(name)))

    def reset(self):
        self.raw = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def summarize(self):
        summary = {}
        for name, x in self.raw.items():
 
            if x.dim() == 2:
                # [B, 2C] expected
                B, D = x.shape
                if D % 2 == 0:
                    C = D // 2
                    scale = x[:, :C]
                    shift = x[:, C:]
                    summary[name] = {
                        "scale_mean_abs": scale.abs().mean(-1).detach().cpu().numpy(),
                        "scale_std": scale.std(-1).detach().cpu().numpy(),
                        "shift_mean_abs": shift.abs().mean(-1).detach().cpu().numpy(),
                        "shift_std": shift.std(-1).detach().cpu().numpy(),
                        'scale': scale.detach().cpu().numpy(),
                        'shift': shift.detach().cpu().numpy()
                    }
                else:
                    summary[name] = {
                        "raw_mean_abs": x.abs().mean(-1).detach().cpu().numpy(),
                        "raw_std": x.std(-1).detach().cpu().numpy(),
                        'scale': scale.detach().cpu().numpy(),
                        'shift': shift.detach().cpu().numpy()
                    }
            else:
                summary[name] = {
                    "raw_mean_abs": x.abs().mean(-1).detach().cpu().numpy(),
                    "raw_std": x.std(-1).detach().cpu().numpy(),
                }
        return summary

class GradCAM1D:
    def __init__(self, model, layer_name):
        self.model = model
        self.modules = dict(model.named_modules())
        self.layer_name = layer_name
        self.activations = None
        self.gradients = None
        self.handles = []

    def register(self):
        layer = self.modules[self.layer_name]

        def fwd_hook(mod, inp, out):
            self.activations = out

        def bwd_hook(mod, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.handles.append(layer.register_forward_hook(fwd_hook))
        self.handles.append(layer.register_full_backward_hook(bwd_hook))

    def reset(self):
        self.activations = None
        self.gradients = None

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def compute(self):
        # expect [B, C, L]
        A = self.activations
        G = self.gradients

        if A.dim() != 3:
            raise ValueError(f"Expected [B,C,L], got {tuple(A.shape)}")

        weights = G.mean(dim=-1, keepdim=True)   # [B, C, 1]
        cam = (weights * A).sum(dim=1)           # [B, L]
        cam = F.relu(cam)

        cam_min = cam.min(dim=-1, keepdim=True)[0]
        cam_max = cam.max(dim=-1, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach().cpu().numpy()

# For choosing the target scalar
#target = out.pow(2).mean() # -> total denoising energy
#target = out[:, waypoint_idx, :].pow(2).mean() # -> one waypoint
#target = out[:, waypoint_idx, coord_idx].mean() # -> one coordinate at one waypoint
#target = -((pred_x0 - gt_x0) ** 2).mean() # -> error to GT noiseless trajectory

class ProbeStore:
    def __init__(self):
        self.data = {}
        self.handles = []
    
    def clear(self):
        self.data = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
    
    def add_forward_hook(self, module, name, detach=True):
        def hook(mod, inp, out):
            self.data[name] = {
                'input': tuple(
                    x.detach() if (detach and torch.is_tensor(x)) else x
                    for x in inp
                ),
                "output": out.detach() if (detach and torch.is_tensor(out)) else out
            }
        self.handles.append(module.register_forward_hook(hook))
    
    def add_backward_hook(self, module, name, detach=True):
        def hook(mod, grad_input, grad_output):
            gout = grad_output[0]
            self.data[name] = gout.detach() if (detach and torch.is_tensor(gout)) else gout
        self.handles.append(module.register_full_backward_hook(hook))
    
def get_module_dict(model):
    return dict(model.named_modules())

def print_matching_modules(model, keywords = ("attn", "film", "up", "mid", "conv")):
    for name, module in model.named_modules():
        low = name.lower()
        if any(k in low for k in keywords):
            print(f"{name:60s} {type(module).__name__}")
