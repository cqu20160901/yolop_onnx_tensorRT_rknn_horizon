# yolop_onnx_tensorRT_rknn

yolop 部署版本，后处理用python语言和C++语言形式进行改写，便于移植不同平台（onnx、tensorRT、rknn）。

# 文件夹结构说明

yolop_onnx：onnx模型、测试图像、测试结果、测试demo脚本

yolop_TensorRT：TensorRT版本模型、测试图像、测试结果、测试demo脚本、onnx模型、onnx2tensorRT脚本(tensorRT-7.2.3.4)

yolop_rknn：rknn模型、测试（量化）图像、测试结果、onnx2rknn转换测试脚本

[onnx 模型路径](https://github.com/cqu20160901/yolop_onnx_tensorRT_rknn/releases/tag/v1.0.0)

# 测试结果

![image](https://github.com/cqu20160901/yolop_onnx_tensorRT_rknn/blob/main/yolop_onnx/onnx_result.jpg)

说明：Focus 层用一个卷积层进行了替换，激活函数 Hardswish 用 Relu 进行了替换， 实际上板端芯片还需要考虑 SPP 或 ASPP 中 pool 层的 kernel_size，常用板端芯片的 NPU 对 pool 或 conv 的 kernel_size最大支持7，大于7的用CPU支持，这样会影响模型的效率。由于用了一部分训练数据和迭代的次数不多，效果并不是很好。仅供测试流程用。

