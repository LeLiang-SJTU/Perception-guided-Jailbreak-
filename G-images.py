import csv
import atexit
from methods.OurMethod import GsafeSetence
from methods.LLMapi import Gimages
import pandas as pd

def generate_images(df, image_models=["dalle-3","dalle-2" "ali", "tencent", "zhipu"], output_path):
    """对每个提示词使用不同模型生成图像"""
    # 使用传入的output_path创建checkpoint路径
    checkpoint_path = output_path.replace('.csv', '_checkpoint.csv')
    
    for model in image_models:
        print(f"\n使用 {model} 模型生成图像...")
        
        # 为每个模型创建新的结果列
        df[f'{model}_image_url'] = ''
        df[f'{model}_image_path'] = ''
        df[f'{model}_status'] = ''
        df[f'{model}_info'] = ''
        
        for idx, row in df.iterrows():
            if pd.isna(row['direct_prompt']):
                continue
                
            print(f"处理第 {idx} 行...")
            try:
                # 调用Gimages函数生成图像
                result = Gimages(
                    prompt=row['direct_prompt'],
                    model_type=model,
                    path=f"images/{model}",
                    filename=f"{idx}_{model}.png"
                )
                
                # 保存结果
                if result[1]:  # 如果生成成功
                    df.at[idx, f'{model}_image_url'] = result[1]
                    df.at[idx, f'{model}_image_path'] = result[23][1]
                    df.at[idx, f'{model}_status'] = 'success'
                    df.at[idx, f'{model}_info'] = str(result[3])
                else:
                    df.at[idx, f'{model}_status'] = 'failed'
                    df.at[idx, f'{model}_info'] = str(result[3])
                
                # 每处理10行保存检查点
                if idx % 10 == 0:
                    df.to_csv(checkpoint_path, index=False)
                    print(f"检查点保存到第 {idx} 行")
                    
            except Exception as e:
                print(f"处理第 {idx} 行时发生错误: {str(e)}")
                df.at[idx, f'{model}_status'] = 'error'
                df.at[idx, f'{model}_info'] = str(e)
                continue
    
    # 最终结果保存到原始output_path
    return df

def main(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 如果direct_prompt列不存在，先生成安全提示词
    if 'direct_prompt' not in df.columns:
        print("生成安全提示词...")
        results = []
        for sentence in df['oriSentence']:
            safe_result = GsafeSetence(sentence)
            results.append(safe_result)
        
        df['direct_prompt'] = [result['output'] for result in results]
        df['direct_prompt_info'] = [result for result in results]
    
    # 生成图像
    output_path = file_path.replace('.csv', '_with_images.csv')
    df = generate_images(df, output_path)
    df.to_csv(output_path, index=False)
    print(f"处理完成，最终结果已保存至: {output_path}")

if __name__ == '__main__':
    unsaftySentence_file_path = 'datas/GPT3.5DALLE3/gptdalle3_prompts.csv'
    main(unsaftySentence_file_path)