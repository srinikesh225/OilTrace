# Third-Party Licenses

OILTRACE's optional model segmentation backend (`SEGMENTER_BACKEND=model`,
`backend/app/stages/segmenters/model_segmenter.py`) uses a pretrained
ResNet50 DeepLabV3+ oil-spill segmentation model and its preprocessing code
from the project below. The pretrained weights are **not** distributed with
OILTRACE and are not committed to this repository.

---

## HTSM Oil Spill Segmentation

- **Project:** Oil Spill Segmentation using Deep Encoder-Decoder models
- **Authors:** Abhishek Ramanathapura Satyanarayana, Maruf A. Dhali
- **Repository:** https://github.com/AbhishekRS4/HTSM_Oil_Spill_Segmentation
- **Model:** ResNet50 DeepLabV3+ (classes: sea_surface, oil_spill,
  oil_spill_look_alike, ship, land)
- **Weights:** `oil_spill_seg_resnet_50_deeplab_v3+_80.pt`, obtained from the
  authors' Hugging Face Space
  (https://huggingface.co/spaces/abhishekrs4/Oil_Spill_Segmentation)
- **Paper:** A. R. Satyanarayana and M. A. Dhali, "Oil Spill Segmentation Using
  Deep Encoder-Decoder Models," Proceedings of ICPRAM 2025, pp. 741-748,
  SciTePress. doi:10.5220/0013259600003905
- **License:** MIT

### MIT License

```
MIT License

Copyright (c) 2023 Abhishek Ramanathapura Satyanarayana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The original license terms are preserved unchanged. See the upstream repository
for the authoritative `LICENSE.md`.
