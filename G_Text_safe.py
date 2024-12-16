import csv
from methods.LLMapi import LLMmodel
import atexit
from methods.OurMethod import GsafeSetence,save_as_csv




if __name__ == '__main__':


    unsaftySentence_file_path ='datas/GPT3.5DALLE3/gptdalle3_prompts.csv'
    # 打开CSV文件
    with open(unsaftySentence_file_path, 'r',encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]

    # 程序终止时调用
    atexit.register(save_as_csv, rows, unsaftySentence_file_path)


    if len(rows[0]) < 7:
        rows[0].extend(["direct_prompt",  "direct_prompt_info"])
        # 假设我们有一个名为 data 的列表，包含着1000个元素


    # 用我们的方法产生图像
    for i in range(1, 1001):
        if len(rows[i]) < 7:
            rows[i].extend(["direct_prompt",  "direct_prompt_info"])

        prompt = GsafeSetence(rows[i][2])
        result = prompt["output"]
        rows[i][6] = result
        rows[i][7] = prompt
      








