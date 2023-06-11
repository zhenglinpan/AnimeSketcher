# AnimeSketcher

**This project is on going and will be updated.**
***@Date:Jun 11 2023***

## Clarification
I would like to emphasize my utmost respect for the diligent efforts of animation producers and the valuable results they create. The project AnimeSketcher is solely intended for personal research purposes and is in no way associated with any commercial endeavors. Any images or materials labeled as "original" within the project are strictly used for demonstration and will be clearly identified with their original sources. The copyrights of these materials rightfully belong to their respective owners, and should they request removal, the material will be promptly deleted without delay.


### Same scene different cut
<p align="center">
  <img src="https://github.com/ZhenglinPan/AnimeSketcher/blob/master/others/img03.gif" width="" alt="accessibility text">
</p>

### Comparision with canny
<p align="center">
  <img src="https://github.com/ZhenglinPan/AnimeSketcher/blob/master/others/img01.jpg" width="" alt="accessibility text">
</p>

### Different scene, different after-effect
<p align="center">
  <img src="https://github.com/ZhenglinPan/AnimeSketcher/blob/master/others/img02.jpg" width="" alt="accessibility text">
</p>

* images from 進撃の巨人 by WIT Studio *

This project uses deep learning methods to transform final films of an animation back into sketchs.

本项目的目的是使用深度学习的方法把动画的最终画面变换回原画。

Currently the model is implemented with cycleGAN and a pretrain model can be downloaded [here](https://drive.google.com/file/d/1NwKzV5UxqBrgXHCXa_r6WzJcV8XbRlNO/view?usp=sharing).

Dataset is not able to be made public due to potential copyright issues, unfortunately, but you can test the model provided or train your own model instead with implemented code, in such case, a few work need to be done on the code.

Test the model with
```python
python test.py --mode test --patch 4
```

