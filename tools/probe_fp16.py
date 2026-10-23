"""Quick probe of fp16 model dtype behaviour on this machine."""
import copy
import sys

import torch

sys.path.insert(0, ".")

from src.utils.checkpoints import load_checkpoint
from src.models.temporal_tcn import TemporalTCN, TCNConfig

ckpt = load_checkpoint("results/raw/tcn_split01_seed42/model_calibrated.pt", map_location="cpu")
model = TemporalTCN(TCNConfig.from_dict(ckpt.meta["model_config"]))
model.load_state_dict(ckpt.model)
model.eval()
half_model = copy.deepcopy(model).half()
x = torch.randn(2, 16, 16).half()

print("param dtypes:", {n: p.dtype for n, p in model.named_parameters()})
print("half param dtype sample:", next(half_model.parameters()).dtype)
try:
    with torch.no_grad():
        out = half_model(x)
    print("CPU half forward OK", type(out))
except RuntimeError as exc:
    print("CPU half forward FAILED:", exc)

if torch.cuda.is_available():
    print("cuda available")
    g = half_model.cuda()
    xg = x.cuda()
    print("gpu param dtype sample:", next(g.parameters()).dtype, next(g.parameters()).device)
    print("gpu bias dtype:", g.convs[0].bias.dtype if hasattr(g, "convs") else "?")
    for name, param in g.named_parameters():
        print("  ", name, param.dtype, param.device)
        break
    try:
        with torch.no_grad():
            out = g(xg)
        print("CUDA half forward OK")
    except RuntimeError as exc:
        print("CUDA half forward FAILED:", exc)
else:
    print("cuda NOT available")
