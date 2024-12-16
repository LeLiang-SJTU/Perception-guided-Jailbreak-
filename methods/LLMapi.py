from openai import OpenAI
import qianfan
import json
import sys
from http import HTTPStatus
import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from dashscope import ImageSynthesis
import dashscope
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
from zhipuai import ZhipuAI
import csv

#阿里云接口调用
dashscope.api_key  = ''

#百度Ai接口
os.environ["QIANFAN_ACCESS_KEY"] = ""
os.environ["QIANFAN_SECRET_KEY"] = ""

#openai接口
openaikey = ""
openaiurl = 'https://api.openai.com/v1'

#腾讯云接口
SecretId = ''
SecretKey = ''

#智普接口
zhipu_api_key=""

#保存数据
def save_as_json(data, directory ,file_name):
    if not os.path.exists(directory ):
        os.makedirs(directory )

    file_path = os.path.join(directory, file_name)
    print(file_path)
    # 将数据写入 JSON 文件
    with open(file_path, 'w',encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"数据已保存到 {file_path}")

def save_as_csv(datas,file_path):
    # 将修改后的数据写入新的CSV文件
    if not os.path.exists(folder_path ):
        os.makedirs(folder_path )

    with open(file_path, 'w', newline='',encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(datas)
    print(f"数据保存成功{file_path}")



#大模型调用
def LLMmodel(input,model="gpt-4o"):
    print(f"运行的模型是{model}")
    if "gpt" in model:
        client = OpenAI(
            # This is the default and can be omitted
            base_url=openaiurl,
            api_key=openaikey
        )

        response = client.chat.completions.create(
            model=model,
            messages=input
        )

        result = response.choices[0].message.content

    elif 'baidu' in model :
        baidu = qianfan.ChatCompletion()
        print(model)
        print(model[6:])
        #response = baidu.do(endpoint=model[6:], messages= input )
        response = baidu.do(model="ERNIE-4.0-Turbo-8K", messages= input )


        #response = qianfan.ChatCompletion().do(endpoint=model, messages=input, temperature=0.95, top_p=0.8, penalty_score=1,disable_search=False, enable_citation=False)

        result = response.body['result']
        print(f"是百度在运行，结果是{result}")

    elif "ali" in model:
        response = dashscope.Generation.call(
            model=dashscope.Generation.Models.qwen_turbo,
            messages=input,
        )
        # 如果调用成功，则打印模型的输出
        if response.status_code == HTTPStatus.OK:
            print("调用了阿里模型")
            result = response.output.text
        # 如果调用失败，则打印出错误码与失败信息
        else:
            return  None


        #result = ""
    else:
        #print("llama模型暂不支持")
        sys.exit("模型输入错误")


    return result



#下载保存图片
def download_image(url, folder_path, file_name):
    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 设置重试策略
    retry_strategy = Retry(
        total=5,  # 总共尝试的次数
        status_forcelist=[429, 500, 502, 503, 504],  # 定义因指定的状态码需要重试的情况
        method_whitelist=["HEAD", "GET", "OPTIONS"],  # 指定哪些方法需要重试
        backoff_factor=1  # 指定等待时间
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        # 发送 GET 请求获取图片
        response = session.get(url, timeout=10)  # 增加超时以防无限等待
        response.raise_for_status()  # 直接抛出异常，如果状态码不是200

        # 拼接文件路径
        file_path = os.path.join(folder_path, file_valid_name(file_name))
        # 保存图片
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"图片已保存到 {file_path}")
        return file_path

    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        return None

def file_valid_name(name):
    """
    清理文件名中不合法的字符
    """
    return "".join(c for c in name if c not in '\\/:*?"<>|')



    

#dalle 调用
def ImageCreat(input, model='dall-e-3',size="1024x1024",n =1,quality="standard",style="natural",response_format="url"):
    client = OpenAI(
        base_url=  openaiurl,
        api_key = openaikey
    )

    try:
        # 可能触发异常的代码块
        image = client.images.generate(
            model=model,
            prompt=input,
            n=n,
            quality=quality,
            size=size,
            style=style
        )
        print(image.created)

        print(image.data[0].url)


        return True,image

    except Exception as e:
        print(f"图像生成错误: {e}")

        return False,str(e)


#调用阿里云接口
def ImageCreatAliTY(input,model=ImageSynthesis.Models.wanx_v1):
    print(f"开始生成{input}的图像")
    try:
        # 可能触发异常的代码块
        #prompt = 'Mouse rides elephant'
        rsp = ImageSynthesis.call(model=model,
                                  prompt=input,
                                  n=1,
                                  size='1024*1024')
        print('Success, rsp: %s' % rsp)
        print('output: %s' % rsp.output)
        print('usage: %s' % rsp.usage)
        if rsp.usage == None:
            return False, str(rsp)
        else:
            return True, rsp
    except Exception as e:
        print(f"图像生成错误: {e}")
        return False, str(e)


#调用腾讯云接口

def ImageCreatTX(input):

    try:
        # 实例化一个认证对象，入参需要传入腾讯云账户 SecretId 和 SecretKey，此处还需注意密钥对的保密
       
        cred = credential.Credential(SecretId, SecretKey)
        # 实例化一个http选项，可选的，没有特殊需求可以跳过
        httpProfile = HttpProfile()
        httpProfile.endpoint = "hunyuan.tencentcloudapi.com"

        # 实例化一个client选项，可选的，没有特殊需求可以跳过
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        # 实例化要请求产品的client对象,clientProfile是可选的
        client = hunyuan_client.HunyuanClient(cred, "ap-guangzhou", clientProfile)

        # 实例化一个请求对象,每个接口都会对应一个request对象
        req = models.TextToImageLiteRequest()
        params = {
            "Prompt": input,
            "RspImgType": "url"

        }
        req.from_json_string(json.dumps(params))

        # 返回的resp是一个TextToImageLiteResponse的实例，与请求对象对应
        resp = client.TextToImageLite(req)
        result = resp.to_json_string()
        # 输出json格式的字符串回包
        print(result)
        return True,resp

    except TencentCloudSDKException as e:

        print(f"图像生成错误: {e}")

        return False,str(e)

#智普接口调用 
def ImageCreatZP(input):

    try:

        client = ZhipuAI(api_key=zhipu_api_key)  # 请填写您自己的APIKey

        response = client.images.generations(
            model="cogview-3",  # 填写需要调用的模型编码
            prompt=input,
        )

        print(response.data[0].url)
        # 检查响应状态码
        print("图像生成成功")



        return True,response


    except Exception as e:
        print(f"图像生成错误: {e}")

        return False,str(e)


