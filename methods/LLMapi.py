# 标准库导入
import json
from datetime import datetime
from pathlib import Path
import time

# 第三方库导入
import toml
import pandas as pd
import requests

# OpenAI相关
from openai import OpenAI

# 百度文心相关


# 智谱AI相关
import zhipuai

# 阿里云相关
import dashscope
from dashscope import ImageSynthesis  # 添加这行

# 腾讯云相关
from tencentcloud.common import credential
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models as hunyuan_models

# 加载配置文件
def load_config():
    config_path = Path(__file__).parent.parent / 'config.toml'
    if not config_path.exists():
        raise FileNotFoundError("配置文件不存在，请确保config.toml文件在正确的位置")
    return toml.load(config_path)

def save_config(config):
    """保存配置到文件"""
    config_path = Path(__file__).parent.parent / 'config.toml'
    with open(config_path, 'w', encoding='utf-8') as f:
        toml.dump(config, f)

# 初始化配置
config = load_config()
API_KEYS = config['api_keys']
MODEL_CONFIG = config['model_config']
SYSTEM_CONFIG = config['system']

# 确保保存目录存在
Path(SYSTEM_CONFIG['image_save_path']).mkdir(exist_ok=True)
Path(SYSTEM_CONFIG['csv_save_path']).mkdir(exist_ok=True)

def check_api_key(config, model_type):
    """检查特定模型的API密钥是否已配置"""
    api_keys = config['api_keys']
    
    if model_type == "openai":
        return bool(api_keys.get('openai'))
    elif model_type == "zhipu":
        return bool(api_keys.get('zhipu'))
    elif model_type == "ali":
        return bool(api_keys.get('ali'))
    elif model_type == "tencent":
        return bool(api_keys.get('tencent_secret_id')) and bool(api_keys.get('tencent_secret_key'))
    
    return False

def update_api_key(config, model_type, api_key):
    """更新API密钥配置"""
    if model_type == "tencent":
        key_id, key_secret = api_key.split(',')
        config['api_keys']['tencent_secret_id'] = key_id.strip()
        config['api_keys']['tencent_secret_key'] = key_secret.strip()
    else:
        config['api_keys'][model_type] = api_key
    
    save_config(config)
    return config


#保存生成记录
def save_as_csv(prompt, model_type, status, url=None, error=None):
    """保存生成记录到CSV"""
    log_file = Path(SYSTEM_CONFIG['csv_save_path']) / 'new.csv'
    data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prompt': prompt,
        'model': model_type,
        'status': status,
        'url': url,
        'error': error
    }
    df = pd.DataFrame([data])
    
    if log_file.exists():
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, index=False)

def download_image(url, model_type, path, filename):
    """下载生成的图片"""
    try:
        response = requests.get(url, timeout=SYSTEM_CONFIG['timeout'])
        if response.status_code == 200:
            filename = filename or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_type}.png"
            # 创建完整的保存路径
            save_dir = Path(SYSTEM_CONFIG['image_save_path']) / path
            save_dir.mkdir(parents=True, exist_ok=True)  # 创建所有必要的父目录
            save_path = save_dir / filename
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True, str(save_path)
        return False, "下载失败"
    except Exception as e:
        return False, str(e)



#图像生成
def ImageCreatOpenAI(prompt, model=None, **kwargs):
    """OpenAI DALL-E 图像生成
    Args:
        prompt (str): 提示词
        model (str, optional): 模型名称，默认从配置文件读取
        **kwargs: 其他参数，可包括 size, quality, style, n 等
    """
    try:
        client = OpenAI(api_key=API_KEYS['openai'])
        model = model or MODEL_CONFIG['openai_default']
        
        # 合并默认参数和用户传入的参数
        params = MODEL_CONFIG.get('model_params', {}).get('openai', {}).copy()
        params.update(kwargs)
        params.update({
            "model": model,
            "prompt": prompt,
        })
        
        response = client.images.generate(**params)
        save_as_csv(prompt, "openai", "success", response)
        return True, response
    except Exception as e:
        save_as_csv(prompt, "openai", "error", error=str(e))
        return False, str(e)

def ImageCreatZP(prompt, model=None, **kwargs):
    """智谱 AI 图像生成
    Args:
        prompt (str): 提示词
        model (str, optional): 模型名称，默认从配置文件读取
        **kwargs: 其他参数
    """
    try:
        client = zhipuai.ZhipuAI(api_key=API_KEYS['zhipu'])
        model = model or MODEL_CONFIG['zhipu_model']
        
        # 合并默认参数和用户传入的参数
        params = MODEL_CONFIG.get('model_params', {}).get('zhipu', {}).copy()
        params.update(kwargs)
        
        # 使用新的API调用方式
        response = client.images.generations(
            model=model,
            prompt=prompt,
            **params
        )
        print(response)
        if hasattr(response, 'data') and response.data:
            
            return True, response
        
        save_as_csv(prompt, "zhipu", "success", response)
        return False, "未获取到图片URL"
    except Exception as e:
        save_as_csv(prompt, "zhipu", "error", error=str(e))
        return False, str(e)

def ImageCreatTX(prompt, model_type="lite", **kwargs):
    """腾讯云图像生成
    Args:
        prompt (str): 提示词
        model_type (str): 模型类型，可选 'lite' 或 'hunyuan'
        **kwargs: 其他参数
    Returns:
        tuple: (bool, str) - (是否成功, URL或错误信息)
    """
    try:
        cred = credential.Credential(
            API_KEYS['tencent_secret_id'],
            API_KEYS['tencent_secret_key']
        )
        client = hunyuan_client.HunyuanClient(cred, "ap-guangzhou")
        
        if model_type.lower() == "lite":
            req = hunyuan_models.TextToImageLiteRequest()
            req.Prompt = prompt
            req.RspImgType = "url"
            
            response = client.TextToImageLite(req)
            print(response)
            
            if hasattr(response, 'Images') :

                return True, response
            
            save_as_csv(prompt, "tencent_lite", "success", response)
            return False, "获取图片链接失败"
            
        else:
            # 混元生图
            req = hunyuan_models.SubmitHunyuanImageJobRequest()
            req.Prompt = prompt
            submit_response = client.SubmitHunyuanImageJob(req)
            print(submit_response)
            
            if not hasattr(submit_response, 'JobId'):
                return False, "未获取到任务ID"
                
            query_req = hunyuan_models.QueryHunyuanImageJobRequest()
            query_req.JobId = submit_response.JobId
            
            for _ in range(10):  # 尝试10次，可以修改次数
                query_response = client.QueryHunyuanImageJob(query_req)
                print(query_response)
                if query_response.JobStatusMsg == "处理完成":
                    url = query_response.ResultImage[0]
                    save_as_csv(prompt, "tencent_hunyuan", submit_response.JobId, url)
                    return True, url
                else:
                    print(query_response.JobStatusMsg)
                    time.sleep(2)
            
            return False, "生成超时"
            
    except Exception as e:
        save_as_csv(prompt, f"tencent_{model_type}", "error", error=str(e))
        return False, str(e)

#调用阿里云接口
def ImageCreatAliTY(prompt, model=None, **kwargs):
    """阿里云通义万相图像生成
    Args:
        prompt (str): 提示词
        model (str, optional): 模型名称，默认为 'wanx-v1'
        **kwargs: 其他参数
    Returns:
        tuple: (bool, str) - (是否成功, URL或错误信息)
    """
    try:
        dashscope.api_key = API_KEYS['ali']
        model = ImageSynthesis.Models.wanx_v1
        
        # 设置只生成一张图片
        params = {
            'model': model,
            'prompt': prompt,
            'n': 1,  # 明确指定只生成1张图片
            'size': '1024*1024'  # 可选的图片尺寸设置
        }
        params.update(kwargs)  # 允许覆盖默认参数
        
        response = ImageSynthesis.call(**params)
        
        if response.status_code == 200 and response.output.results:
           
            save_as_csv(prompt, "ali", "success", response)
            return True, response
            
        save_as_csv(prompt, "ali", "error", error=response.message)
        return False, response.message
    except Exception as e:
        save_as_csv(prompt, "ali", "error", error=str(e))
        return False, str(e)



#图像生成
def Gimages(prompt, model_type="openai", path=None, filename=None):
    """
    根据不同模型类型调用相应的图像生成API
    
    参数:
    prompt: 图像生成提示词
    model_type: 模型类型 ("openai"/"ali"/"tencent"/"zhipu")
    """
    
    # 根据model_type选择对应的图像生成函数
    if model_type == "openai":
        ifiamge, safeCheck = ImageCreatOpenAI(prompt, model="dall-e-3")
    elif model_type == "ali": 
        ifiamge, safeCheck = ImageCreatAliTY(prompt)
    elif model_type == "tencent":
        ifiamge, safeCheck = ImageCreatTX(prompt)  # 腾讯接口
    elif model_type == "zhipu":
        ifiamge, safeCheck = ImageCreatZP(prompt)  # 智谱接口
    else:
        return [prompt, False, 0, "不支持的模型类型"]

    if ifiamge:
        # 处理成功情况
        try:
            if model_type == "openai":
                createtime = datetime.now().timestamp()
                reversed_prompt = safeCheck.data[0].revised_prompt
                url = safeCheck.data[0].url
            elif model_type == "ali":
                createtime = datetime.now().timestamp()
                reversed_prompt = prompt  # 阿里接口可能没有修改后的prompt
                url = safeCheck.output.results[0].url  # 获取第一张图片的URL
            elif model_type == "tencent":
                createtime = datetime.now().timestamp()
                reversed_prompt = prompt
                url = safeCheck.ResultImage
            elif model_type == "zhipu":
                createtime = datetime.now().timestamp()
                reversed_prompt = prompt
                url = safeCheck.data[0].url
                
            # 下载并保存图片
            image = download_image(url, model_type, path, filename)
            
            return [prompt, True, image, [createtime, reversed_prompt]]
            
        except Exception as e:
            return [prompt, False, 0, f"处理结果时出错: {str(e)}"]
    else:
        return [prompt, False, 0, safeCheck]




#文本生成
def generate_text_with_llm(prompt, model=None, **kwargs):
    """根据模型名称调用不同的语言模型进行文本生成
    Args:
        prompt (str): 提示词
        model (str, optional): 模型名称，可选 'openai'/'gpt-3.5'/'gpt-4'/'baidu'/'ernie'/'ali'/'qwen'/'tencent'/'chatglm' 等
        **kwargs: 其他参数，如temperature、max_tokens等
    Returns:
        tuple: (是否成功, 响应结果)
    """
    try:
        # 获取模型类型
        if not model:
            model = MODEL_CONFIG.get('default_text_model', 'gpt-4o')
            
        model = model.lower()
        
        # OpenAI模型
        if 'gpt' in model or 'openai' in model:
            client = OpenAI(api_key=API_KEYS['openai'])
            params = MODEL_CONFIG.get('model_params', {}).get('openai', {}).copy()
            params.update(kwargs)
            print(params)
            print(model)
            params.update({
                'model': model ,
                'messages': prompt
            })
            response = client.chat.completions.create(**params)
            return True, response.choices[0].message.content
            
            
        # 阿里通义千问
        elif 'ali' in model or 'qwen' in model:
            dashscope.api_key = API_KEYS['ali']
            params = MODEL_CONFIG.get('model_params', {}).get('ali', {}).copy()
            params.update(kwargs)
            params.update({
                'model': 'qwen-turbo',
                'messages': prompt
            })
            response = dashscope.Generation.call(**params)
            if response.status_code == 200:

                return True, response.output.text
            return False, response.message
            
            
        # 智谱ChatGLM
        elif 'chatglm' in model:
            client = zhipuai.ZhipuAI(api_key=API_KEYS['zhipu'])
            params = MODEL_CONFIG.get('model_params', {}).get('zhipu', {}).copy() 
            params.update(kwargs)
            response = client.chat.completions.create(
                model="chatglm_turbo",
                messages=prompt,
                **params
            )
            save_as_csv(prompt, "zhipu", "success", response.choices[0].message.content)
            return True, response.choices[0].message.content
            
        else:
            raise ValueError(f"不支持的模型类型: {model}")
            
    except Exception as e:
        save_as_csv(prompt, model, "error", error=str(e))
        return False, str(e)


if __name__ == "__main__":
    test_prompt = "请用简短的话介绍一下你自己"
    
    # 测试不同的模型
    models = ["gpt-4o", "chatglm", "qwen", "hunyuan","baidu"]
    
    for model in models:
        print(f"\n测试 {model} 模型:")
        success, response = generate_text_with_llm(test_prompt, model=model)
        print(f"成功: {success}")
        print(f"响应: {response}")
        print("-" * 50)



    '''
    
    # 测试所有图像生成模型
    test_prompt = "新年快乐"
    models = ["openai", "zhipu", "tencent", "ali"]
    
    for model in models:
        print(f"\n测试 {model} 模型:")
        response = Gimages(test_prompt, model_type=model)

        print(f"响应: {response}")

        
        print("-" * 50)
        #将返回的结果保存到csv文件中
        save_as_csv(test_prompt, model, "success", response)



    '''
