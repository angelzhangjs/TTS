#!/bin/bash
# remove existing installs just to be safe
pip uninstall -y torch torchvision torchaudio

# reinstall compatible versions together
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# test import
python - <<END
try:
    import torch
    import torchvision
    import torchaudio
    print("Torch:", torch.__version__, "Torchvision:", torchvision.__version__, "Torchaudio:", torchaudio.__version__)
    from torchvision.ops import nms
    print("✅ torchvision::nms exists!")
except Exception as e:
    print("❌ Error:", e)
END


