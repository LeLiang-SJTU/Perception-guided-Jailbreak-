import csv
from methods.LLMapi import ImageCreat, ImageCreatAliTY, ImageCreatTX, ImageCreatZP, save_as_csv,download_image
import atexit
import json



#图像生成
def Gimages(prompt, count, model_type="openai"):
    """
    根据不同模型类型调用相应的图像生成API
    
    参数:
    prompt: 图像生成提示词
    count: 计数器
    model_type: 模型类型 ("openai"/"ali"/"tencent"/"zhipu")
    """
    
    # 根据model_type选择对应的图像生成函数
    if model_type == "openai":
        ifiamge, safeCheck = ImageCreat(prompt, model="dall-e-3")
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
                createtime = safeCheck.created
                reversed_prompt = safeCheck.data[0].revised_prompt
                url = safeCheck.data[0].url
            elif model_type == "ali":
                createtime = safeCheck.request_id
                reversed_prompt = prompt  # 阿里接口可能没有修改后的prompt
                url = safeCheck.output.url
            elif model_type == "tencent":
                createtime = "tencent_" + str(count)
                reversed_prompt = prompt
                url = json.loads(safeCheck.to_json_string())["ImageUrls"][0]
            elif model_type == "zhipu":
                createtime = "zhipu_" + str(count)
                reversed_prompt = prompt
                url = safeCheck.data[0].url
                
            # 下载并保存图片
            image = download_image(
                url, 
                folder_path + f'/Images_{datasori}_{model_type}_{promptmodel}',
                f"{count}_{datasori}_{model_type}_{promptmodel}_{createtime}.png"
            )
            
            return [prompt, True, image, [createtime, reversed_prompt]]
            
        except Exception as e:
            return [prompt, False, 0, f"处理结果时出错: {str(e)}"]
    else:
        return [prompt, False, 0, safeCheck]




if __name__ == '__main__':
    datasori = "GPT"
    promptmodel = "gpt-3.5-turbo"
    
    # 定义要使用的图像生成模型列表
    image_models = ["openai", "ali", "tencent", "zhipu"]
    
    for imagemodel in image_models:
        print(f"当前使用的图像生成模型: {imagemodel}")
        
        # 文件保存地址
        unsaftySentence_file_path = f"datas/{datasori}/{datasori}_{imagemodel}_{promptmodel}.csv"
        # 文件保存路径
        folder_path = f"datas/{datasori}"
        
        # 打开CSV文件
        with open(unsaftySentence_file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = [row for row in reader]
            
        # 程序终止时调用
        atexit.register(save_as_csv, rows, unsaftySentence_file_path)
        
        # 首行名称
        if len(rows[0]) < 14:
            rows[0].extend([""] * (14 - len(rows[0])))
        rows[0][10:14] = [
            f"{imagemodel}_prompt",
            f"{imagemodel}_prompt_passed",
            f"{imagemodel}_prompt_image", 
            f"{imagemodel}_prompt_image_info"
        ]
        
        # 用我们的方法产生图像
        for i in range(3, 1001):
            try:
                if len(rows[i]) < 14:
                    rows[i].extend([""] * (14 - len(rows[i])))
                
                elif rows[i][3] == 'FALSE' and rows[i][10] == '':
                    print(f"使用{imagemodel}模型生成图像中...")
                    
                    prompt = rows[i][6]
                    result = Gimages(prompt, rows[i][0], model_type=imagemodel)
                    rows[i][10:14] = result
                    
                    # 每生成一张图片后保存一次CSV
                    save_as_csv(rows, unsaftySentence_file_path)
                    
                print(f"第{rows[i][0]}行处理完成")
                print(f"提示词: {rows[i][6]}")
                print(f"生成状态: {rows[i][10]}")
                
            except Exception as e:
                print(f"处理第{i}行时发生错误: {str(e)}")
                continue
        
        print(f"{imagemodel}模型的所有图像生成完成")