## FGO刷材料脚本

见fgo文件夹。

贪玩要求:

- Python库依赖: `wisepy win32api win32con win32gui PyQt5 scipy numpy pytorch generic_classifier`

- [从此处下载场景识别模型](https://pan.baidu.com/s/1il853dDb0_aU__5F_VfpFQ), 放于`fgo/fgo_data`里, 命名为`mml`.


- 蓝叠模拟器: 将模拟器的分辨率调整为1280 * 720, 并将模拟器左上角对准屏幕左上角。不能人工修改模拟器窗口，若窗口被resize请reset。

- FGO: 在模拟器中下载并打开游戏, 路线【迦勒底之门】 -> 【每日任务】, 选择一个要刷的图, 放置在下图的红线框内:

![选择任务](./select_task.png)

最后在管理员模式(建议powershell, cmder会蜜汁出错)下, 运行根目录`auto_fgo.py`即可。

如果想要随机刷图，修改码源`fgo/管理室.py`42行, 把

```python

    # if 随机选择:
    # e = 每日任务()
    # yield MoveTo(1260, e.y)
    # yield click()
    # yield Wait(Exact(0.5))
    # yield MoveTo(e.x, e.y)
    # else:
    yield MoveTo(923, 150)

```
改为

```python

    # if 随机选择:
    # e = 每日任务()
    yield MoveTo(1260, e.y)
    yield click()
    yield Wait(Exact(0.5))
    yield MoveTo(e.x, e.y)
    # else:
    # yield MoveTo(923, 150)

```

**注意**: **脚本没有处理体力不够或者刷图失败的情况**。

- 体力用尽后请中止程序恢复体里后再次运行。

- 战斗如果失败, 也请手动停止程序回到【每日任务】处重启脚本。


关于自训练模型
--------------------


你可能觉得我训练的模型在你那边不work, 事实上确实有这个问题: 比如首页英灵是贞德而不是大王时, 我这边就经常识别不出来。

在这个脚本中的机器学习任务是场景识别，仅仅是从模拟器的图像中分辨出当前场景是以下哪一个:

- battle
- not_battle
- battle_ending

你可以从[这个链接](https://pan.baidu.com/s/1xt0HfCAGQ0i4015ETCvwbA)下载我的训练数据, 也可以自己手动截图制作模型。

方法是, 在`fgo/fgo_data`里建立文件夹`cls_<类型label名>`, 至少需要三种label, 即`battle`, `not_battle`, `battle_ending`, 即至少需要下述三个文件夹:

```
- fgo
   - fgo_data
     - cls_battle
        - .. png, jpg

     - cls_not_battle
        - .. png, jpg

     - cls_battle_ending
        - .. png, jpg

   - __init__.py
   - ...
```

如果你觉得脚本效果不好, 可以删除`fgo/fgo_data/mml`, 修改数据文件夹，然后重新运行脚本`auto_fgo.py`, 会自动训练新模型。


## 微信签到

见Deprecated文件夹。




