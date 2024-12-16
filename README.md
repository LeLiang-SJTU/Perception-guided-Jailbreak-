# CodeBase of Perception-guided Jailbreak against Text-to-Image Models
The codebase of article “Perception-guided Jailbreak against Text-to-Image Models”
## 文章链接
https://arxiv.org/html/2408.10848 

## 使用方法

安装所有需要的依赖
填写需要的apikey,见LLMapi.py

### 不安全数据集
+ 自己采集不安全数据集
+ 使用G_Text_unsafe
    直接运行G_Text_unsafe.py文件，注意，根据模型不同，需要多次生成。不同模型限制策略不同，默认使用gpt3.5

### 核心方法
 运行G_Text_safe.py，生成大模型产生的更改后的安全文本（待检测）

### 图片生成检测

 运行G_image.py。调用接口生成图片

### 文本图像相似度检测
使用了BLIp进行检测
