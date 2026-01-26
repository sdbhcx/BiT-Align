# 深度图像生成脚本

本脚本使用Depth-Anything-V2-Small模型为AGD20K数据集生成深度图像。

## 依赖项

- Python 3.7+
- transformers
- PIL

## 安装依赖

```bash
pip install transformers pillow
```

## 使用方法

1. 确保你已经下载了AGD20K数据集
2. 运行以下命令生成深度图像：

```bash
python generate_depth.py --agd20k_root /path/to/AGD20K --output_root /path/to/AGD20K-Depth
```

其中：

- `--agd20k_root` 是AGD20K数据集的根目录
- `--output_root` 是生成的深度图像的输出目录

## 输出结构

生成的深度图像将按照与原始AGD20K数据集相同的目录结构保存：

```
/path/to/AGD20K-Depth/
├── Seen/
│   ├── trainset/
│   │   ├── egocentric/
│   │   │   ├── action1/
│   │   │   │   ├── object1/
│   │   │   │   │   ├── image1_depth.png
│   │   │   │   │   └── ...
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── exo-centric/
│   │       └── ...
│   └── testset/
│       └── ...
└── Unseen/
    └── ...
```


## 深度图像格式

生成的深度图像是单通道灰度图像，其中：

- 较暗的像素表示较近的距离
- 较亮的像素表示较远的距离

## 模型信息

脚本使用 `depth-anything/Depth-Anything-V2-Small-hf` 模型，这是一个基于Transformers架构的高效深度估计模型，具有良好的准确性和推理速度。

该模型来自Hugging Face Hub，可以直接通过transformers库加载使用。

## 注意事项

1. 生成深度图像可能需要较长时间，具体取决于数据集大小和硬件性能
2. 建议在GPU上运行以获得更快的速度
3. 生成的深度图像是估计值，可能与真实深度有一定误差
4. 首次运行时，模型会自动下载到本地缓存
