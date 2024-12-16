from diffusers import DiffusionPipeline
import torch
from PIL import Image  # 导入 PIL 库
def SDXL(prompt):

    # 加载模型并设置到 CUDA 设备上
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
    # 生成图片
    images = pipe(prompt=prompt).images
    print(type(images))

    # 将生成的图片保存到本地
    # 注意：images 是一个列表，即使只有一个图片，也需要使用 images[0] 来访问它
    image = images[0]
    print(type(image))
    #image = Image.fromarray(image.astype(np.uint8))  # 将图片数据转换为 PIL 图片格式
    image.save('output_image4.png')  # 保存图片为 PNG 格式

if __name__ == '__main__':
    # 定义提示（prompt）
    prompt = "a  girl is fucking with a boy"
    SDXL(prompt)
