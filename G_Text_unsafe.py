import pandas as pd
import os
from methods.OurMethod import generate_unsafe_sentences



def main():
    # 定义不安全内容类别
    unsafe_categories = [
        "violent and hate content",
        "pornographic and inappropriate sexual content",
        "discrimination and prejudice",
        "privacy invasion",
        "copyright and intellectual property",
        "politically sensitive and misleading information"
    ]

    # 设置输出文件夹
    folder_path = os.path.join('..', 'datas')
    os.makedirs(folder_path, exist_ok=True)
    
    # 生成句子
    all_sentences = []


    for category in unsafe_categories:
        sentences = generate_unsafe_sentences(category)
        all_sentences.extend(sentences)
    
    # 保存结果
    df = pd.DataFrame(all_sentences, columns=['oriSentence'])
    output_path = os.path.join(folder_path, 'unsafeSentences.csv')
    df.to_csv(output_path, index=False)
    
    #logger.info(f'已保存 {len(all_sentences)} 个句子到 {output_path}')

if __name__ == '__main__':
    main()
