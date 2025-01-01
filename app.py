import streamlit as st
import pandas as pd
import toml
from pathlib import Path
from methods.OurMethod import Our_method
from methods.LLMapi import generate_text_with_llm, Gimages, download_image,update_api_key,check_api_key
import time

# 设置页面配置
st.set_page_config(
    page_title="多模型文生图软件",
    page_icon="🎨",
    layout="wide"
)

# 加载配置
@st.cache_resource
def load_config():
    config_path = Path(__file__).parent / 'config.toml'
    if not config_path.exists():
        st.error("配置文件不存在，请确保config.toml文件在正确的位置")
        return None
    return toml.load(config_path)

def process_jailbreak(prompt, method):
    """处理提示词优化"""
    if method == "安全模式":
        messages = [
            {
                "role": "user",
                "content": f"请将以下图像生成提示词修改为更安全的版本，确保内容适合所有年龄段：\n{prompt}"
            }
        ]
        success, response = generate_text_with_llm(messages)
        return response if success else prompt
        
    elif method == "艺术模式":
        messages = [
            {
                "role": "user",
                "content": f"请将以下图像生成提示词改写成更富有艺术感的描述，加入艺术风格、光影、构图等元素：\n{prompt}"
            }
        ]
        success, response = generate_text_with_llm(messages)
        return response if success else prompt
        
    elif method == "创意模式":
        messages = [
            {
                "role": "user",
                "content": f"请将以下图像生成提示词改写得更有创意和想象力，添加独特的视觉元素和创新概念：\n{prompt}"
            }
        ]
        success, response = generate_text_with_llm(messages)
        return response if success else prompt
        
    elif method == "越狱模式（可以绕过模型安全检查）":
        result = Our_method(prompt)
        print(result)
        return result["output"]

        
    return prompt

def generate_image(prompt, model_type, config, openai_model=None):
    """统一的图像生成函数"""
    # 检查API密钥
    if not check_api_key(config, model_type):
        # 弹出输入框要求输入API密钥
        if model_type == "tencent":
            st.warning("请输入腾讯云API密钥（格式：SecretId, SecretKey）")
            api_key = st.text_input(f"请输入{model_type}的API密钥", type="password")
        else:
            st.warning(f"请输入{model_type}的API密钥")
            api_key = st.text_input(f"请输入{model_type}的API密钥", type="password")
        
        if not api_key:
            return False, "请先配置API密钥"
        
        # 更新配置
        config = update_api_key(config, model_type, api_key)
    
    # 直接使用Gimages函数生成图片
    result = Gimages(prompt, model_type=model_type)
    
    if result[1]:  # 如果生成成功
        return True, result[2]  # 返回图片信息
    else:
        return False, result[3]  # 返回错误信息

def display_image_and_download(success, response, model_type, prompt):
    """显示生成的图片并提供下载选项"""
    if success:
        try:
            if isinstance(response, str):  # 如果response是URL
                st.image(response, caption="生成的图片")
                if st.button("下载图片"):
                    success, path = download_image(response, model_type, "downloads", f"{model_type}_{int(time.time())}.png")
                    if success:
                        st.success(f"图片已保存到: {path}")
                    else:
                        st.error(f"下载失败: {path}")
            else:  # 如果response是图片数据
                st.image(response, caption="生成的图片")
        except Exception as e:
            st.error(f"处理图片时出错: {str(e)}")
    else:
        st.error(f"生成失败: {response}")

def main():
    st.title("多模型文生图软件 🎨")
    
    # 加载配置
    config = load_config()
    if not config:
        st.stop()
    
    # 侧边栏配置
    st.sidebar.title("配置")
    mode = st.sidebar.radio("选择模式", ["单个提示词", "批量处理"])
    
    model_type = st.sidebar.selectbox(
        "选择模型",
        ["openai", "ali", "tencent", "zhipu"],
        help="选择要使用的AI模型"
    )
    
    # OpenAI模型选择
    openai_model = None
    if model_type == "openai":
        openai_model = st.sidebar.selectbox(
            "选择DALL-E版本",
            config['model_config']['openai_models'],
            index=config['model_config']['openai_models'].index(config['model_config']['openai_default']),
            help="选择要使用的DALL-E模型版本"
        )
    
    if mode == "单个提示词":
        st.header("单个提示词生成")
        
        # 输入区
        col1, col2 = st.columns([3, 1])
        with col1:
            prompt = st.text_area("输入提示词", height=100)
        with col2:
            use_jailbreak = st.checkbox("使用提示词优化")
            if use_jailbreak:
                jailbreak_method = st.selectbox(
                    "优化方式",
                    ["安全模式", "艺术模式", "创意模式","越狱模式（可以绕过模型安全检查）"]
                )
                if st.button("预览优化效果", type="primary"):
                    if prompt:
                        optimized_prompt = process_jailbreak(prompt, jailbreak_method)
                        st.info(f"优化后的提示词: {optimized_prompt}")
                    else:
                        st.warning("请先输入提示词")
        
        # 生成按钮
        if st.button("生成图片", type="primary"):
            if prompt:
                with st.spinner("生成中..."):
                    if use_jailbreak:
                        prompt = process_jailbreak(prompt, jailbreak_method)
                        st.info(f"实际使用的提示词: {prompt}")
                    
                    success, response = generate_image(prompt, model_type, config, openai_model)
                    display_image_and_download(success, response, model_type, prompt)
            else:
                st.warning("请输入提示词")
    
    else:  # 批量处理模式
        st.header("批量提示词处理")
        
        uploaded_file = st.file_uploader(
            "上传文件 (支持 .txt, .csv, .xlsx)",
            type=['txt', 'csv', 'xlsx'],
            help="文件应包含一个名为'prompt'的列，或每行一个提示词"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.type == "text/plain":
                    prompts = uploaded_file.read().decode().splitlines()
                    df = pd.DataFrame(prompts, columns=['prompt'])
                elif uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                else:  # Excel
                    df = pd.read_excel(uploaded_file)
                
                st.write("预览数据:")
                st.dataframe(df.head())
                
                use_jailbreak = st.checkbox("批量使用提示词优化")
                if use_jailbreak:
                    jailbreak_method = st.selectbox(
                        "优化方式",
                        ["安全模式", "艺术模式", "创意模式"]
                    )
                
                if st.button("开始批量处理", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, row in df.iterrows():
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        
                        prompt = row['prompt']
                        status_text.text(f"处理中 {idx+1}/{len(df)}: {prompt[:50]}...")
                        
                        if use_jailbreak:
                            prompt = process_jailbreak(prompt, jailbreak_method)
                        
                        success, response = generate_image(prompt, model_type, config, openai_model)
                        if success:
                            st.success(f"成功生成: {prompt[:50]}...")
                        else:
                            st.error(f"生成失败: {prompt[:50]}... - {response}")
                    
                    st.success("批量处理完成！")
                    
            except Exception as e:
                st.error(f"处理文件时出错: {str(e)}")

if __name__ == "__main__":
    main() 