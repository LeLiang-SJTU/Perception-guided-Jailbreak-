import atexit
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForImageTextRetrieval
import os
import pandas as pd



def getscores(img_path, text):
    print("______________________")
    print(img_path)
    print(text)

    processor = BlipProcessor.from_pretrained("../models/blip-itm-base-coco")
    model = BlipForImageTextRetrieval.from_pretrained("../models/blip-itm-base-coco")
    img = Image.open(img_path).convert('RGB')

    # 截断输入文本到512个token以内
    inputs = processor(img, text, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        itm_scores = model(**inputs)[0]
        cosine_score = model(**inputs, use_itm_head=False)[0]
    print(cosine_score.numpy()[0][0])

    # 将itm_scores转换为列表，将cosine_score转换为单个数字
    itm_scores_list = itm_scores.detach().numpy().tolist()[0]
    cosine_score_value = cosine_score.detach().numpy()[0][0]

    return itm_scores_list, cosine_score_value



def process_row(row):
    try:
        if row[image_path] != '0' or row[image_path] != '' and not pd.isnull(row[f'{image_path[:3]}-{textprompt[:3]}-itm_scores_tar']):
            print(f"第{row['id']}")
            return pd.Series(getscores(row[image_path], row[textprompt]))
        else:
            return pd.Series([row[f'{image_path[:3]}-{textprompt[:3]}-itm_scores_tar'], row[f"{image_path[:3]}-{textprompt[:3]}_cosine_score_tar"]])

    except Exception as e:
        print(f"Error processing row: {e}")
        return pd.Series([0, 0])


if __name__ == '__main__':
    # 定义文件夹路径
    folder_path = '../datas/GPT/'

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        # 检查文件是否为CSV文件
        if file_name.endswith('.csv'):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, file_name)

            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                # 打印文件名和DataFrame的前几行
                print(f'读取文件: {file_name}')
                print(df.head())
                print('\n')


                atexit.register(df.to_csv, file_path, index=False)

                textprompt = 'ori_prompt'
                image_path = df.columns[12]
                print(f"image_path:{image_path}")


                if f'{image_path[:3]}-{textprompt[:3]}-itm_scores_tar' in df.columns:
                    print(f"文件已处理: {file_name}",f'{image_path[:3]}-{textprompt[:3]}-itm_scores_tar')
                    continue

                # 应用函数并添加新列
                df[[f'{image_path[:3]}-{textprompt[:3]}-itm_scores_tar',
                    f'{image_path[:3]}-{textprompt[:3]}-cosine_score_tar']] = df.apply(process_row, axis=1)
                df.to_csv(file_path, index=False)
                print(f"保存文件: {file_name}")
            except Exception as e:
                print(f"BIlip失败: {e}——{file_name}")
                with open("../BLIP.txt", "a") as file:
                    file.write(f"BIlip失败: {e}——{file_name}\n")
