import csv
import atexit
from methods.OurMethod import GsafeSetence,save_as_csv
import pandas as pd

def main(file_path):
    # 打开CSV文件
    df = pd.read_csv(file_path)
    # 获取oriSentence列数据
    sentences = df['oriSentence'].tolist()
    
    # 对每个句子应用GsafeSetence方法并保存结果
    results = []
    for sentence in sentences:
        safe_result = GsafeSetence(sentence)
        results.append(safe_result)
    
    # 将结果添加为新列
    df['direct_prompt'] = [result['output'] for result in results]
    df['direct_prompt_info'] = [result for result in results]
    
    # 保存处理后的结果到新的CSV文件
    output_path = file_path.replace('.csv', '_processed.csv')
    df.to_csv(output_path, index=False)
    print(f"处理完成，结果已保存至: {output_path}")

if __name__ == '__main__':
    unsaftySentence_file_path = 'datas/GPT3.5DALLE3/gptdalle3_prompts.csv'
    main(unsaftySentence_file_path)







