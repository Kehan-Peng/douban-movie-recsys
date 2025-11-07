import os
import requests
from dotenv import load_dotenv

# 自动加载项目根目录的 .env 文件
load_dotenv()

class LLMClient:
    def __init__(self, model=None, api_key=None, base_url=None):
        # 优先传入参数，否则读取环境变量
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        self.base_url = base_url or os.getenv("SILICONFLOW_BASE_URL")
        self.model = model or os.getenv("SILICONFLOW_MODEL")

        # 校验必填配置
        if not self.api_key:
            raise ValueError("错误：未设置 SILICONFLOW_API_KEY")
        if not self.base_url:
            raise ValueError("错误：未设置 SILICONFLOW_BASE_URL")
        if not self.model:
            raise ValueError("错误：未设置 SILICONFLOW_MODEL")

    def generate(self, prompt, temperature=0.3, max_tokens=1024):
        """
        调用硅基流动大模型 API
        :param prompt: 用户输入的提示词
        :param temperature: 随机性，0~1
        :param max_tokens: 最大生成长度
        :return: 模型返回的文本结果
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是电影语义标注专家，只输出英文标签，精准、简洁、无多余内容。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.7,
            "stream": False
        }

        try:
            api_url = f"{self.base_url.rstrip('/')}/chat/completions"
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        except requests.exceptions.HTTPError as e:
            return f"HTTP错误: {response.status_code} | {response.text}"
        except Exception as e:
            return f"API调用失败: {str(e)}"


# 全局单例，项目里直接导入这个 llm_client 即可使用
llm_client = LLMClient()