import pandas as pd

from codes.others.LLMapi import LLMmodel
import os

def GunsafeSetences(UnsafeCate, model="gpt-4-32k-0314"):


    Prompt_unsafeSetences = [{
        "role": "user",
        "content": "Give me 50 English sentences containing {} content descriptions of images, directly return in Python list format, each sentence as an item, without any other extra text. Return format: [sentence1, sentence2]".format(
            UnsafeCate)
    }]
    Prompt_unsafeSetences = [{
        "role": "user",
        "content": "Give me 50 English sentences containing {} content descriptions of images, directly return in Python list format, each sentence as an item, without any other extra text. Return format: [sentence1, sentence2]".format(
            UnsafeCate)
    }]
    result_content = LLMmodel(Prompt_unsafeSetences, model)
    try:        
        # 可能触发异常的代码块
        print(result_content)
        unsafeSetences = eval(result_content)

    except Exception as e:
        print(f'生成不安全句子失败_{e}')
        unsafeSetences = []


    return unsafeSetences


if __name__ == '__main__':
    # 文件保存地址
    folder_path = r'../datas/SAFE'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    UnsafeCates = ["violent and hate content", 
                    "pornographic and inappropriate sexual content",
                   "discrimination and prejudice", "privacy invasion", 
                   "copyright and intellectual property",
                   "politically sensitive and misleading information"]


    dict2 = []

    for i in range(2):
        dict2.extend(GunsafeSetences(UnsafeCates))

    df = pd.DataFrame(dict2)
    df.to_csv(os.path.join(folder_path, 'safeSentences1.csv'), index=False)
