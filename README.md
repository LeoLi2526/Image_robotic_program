# 轨迹识别原型

基于图片中的两条黑色边界带，提取中间白色通道的中心轨迹。

## 文件

- `trajectory_recognition.py`：主程序
- `requirements.txt`：依赖
- `trade_original_image.jpg`：输入图片

## 安装

当前实现已按 conda 环境 `image` 设计与验证。

建议使用 Python 3.10+。

安装依赖：

- `opencv-python`
- `numpy`

## 运行

在当前目录执行：

`/home/leoli/anaconda3/envs/image/bin/python trajectory_recognition.py --image trade_original_image.jpg --outdir output`

## 输出

程序会生成：

- `output/01_gray.png`：灰度图
- `output/02_black_mask.png`：黑色区域掩膜
- `output/03_bands_overlay.png`：检测到的上下黑带
- `output/04_centerline_overlay.png`：通道中心轨迹与黑带内侧边线拟合可视化
- `output/05_warped_overlay.png`：透视校正后的内侧边线与中线可视化
- `output/result.json`：中心线点、黑带内侧边线点、拟合参数、偏移量、角度等结果

终端会输出简要摘要，完整点集保存在 `output/result.json`。

## 方法说明

1. 灰度化与高斯滤波
2. 自适应阈值提取黑色带状区域
3. 形态学闭运算连接黑带
4. 连通域筛选出两个主要横向黑带
5. 提取上黑带下边缘、下黑带上边缘
6. 对两条内侧边缘分别做鲁棒直线拟合
7. 基于两条内侧边线构造透视校正区域
8. 在校正后的视角中重新拟合内侧边线与中线
9. 输出横向偏差、轨迹角度和校正前后的边线参数
