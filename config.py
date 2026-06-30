import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# DeepSeek配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


# 数据保存路径
DATA_DIR = "stock_data"

# 日志配置 
LOG_LEVEL = "INFO"