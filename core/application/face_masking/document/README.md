# Sticker数据制作

# 眼罩数据制作

1. 图像RGBA
2. 双眼位置。以下图为例，**两个红色点**的xy坐标分别是(857,1735), (1683, 1735)
3. 构造的参数MaskingOption

```python
import cv2
import numpy as np
from libmasking import *
sticker = cv2.imread('/path/to/sticker_retro.png', cv2.IMREAD_UNCHANGED)
points = np.array([[857, 1735], [1683, 1735]], dtype=np.int32)
parameters = dict(bgr=sticker, eyes_center=points)
masking_option = MaskingOption(MaskingOption.MaskingOption_Sticker, parameters)
```



<table>
    <tr>
        <td align="center"><img src="./sticker_retro.png" alt="showcase" height="360" width="640"></td>
        <td align="center"><img src="./obama-sticker_retro.png" alt="showcase" height="360" width="640"></td>
    </tr>
</table>

