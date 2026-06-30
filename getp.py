#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import time
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import openai
import levistock as lk

# ================== 配置 ==================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DATA_DIR = "./data"

# ================== 工具函数 ==================
def ensure_dir():
    """确保数据目录存在"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def safe_str_to_date(s, fmt="%Y%m%d"):
    """将字符串转为datetime，支持多种格式"""
    if not s:
        return None
    # 尝试不同格式
    formats = ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]
    for f in formats:
        try:
            return datetime.strptime(s, f)
        except:
            continue
    raise ValueError(f"无法解析日期: {s}")

def date_to_str(d, fmt="%Y%m%d"):
    return d.strftime(fmt)

# ================== Agent 核心 ==================
class StockAgent:
    def __init__(self, api_key=DEEPSEEK_API_KEY):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.system_prompt = """
你是一个A股数据助手，负责解析用户的自然语言请求，提取结构化参数。

用户会输入类似：“20210101-20260101半导体资金流向” 或 “白酒板块近一个月资金流向” 等。

你需要提取以下字段并返回 JSON：
- start_date: 起始日期，格式 YYYYMMDD（如果用户未指定，则默认为 30 天前）
- end_date: 结束日期，格式 YYYYMMDD（如果用户未指定，则默认为今天）
- sector_name: 板块名称，例如“半导体”、“白酒”、“新能源”等（必须提取）
- indicator: 指标，目前只支持“资金流向”，默认为“资金流向”

只返回 JSON，不要包含其他内容。
示例输入: "20210101-20260101半导体资金流向"
示例输出: {"start_date":"20210101","end_date":"20260101","sector_name":"半导体","indicator":"资金流向"}
"""
        ensure_dir()

    def parse_request(self, user_input):
        """调用 DeepSeek 解析自然语言"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0,
                max_tokens=200
            )
            content = response.choices[0].message.content.strip()
            # 尝试提取 JSON
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                params = json.loads(match.group())
            else:
                params = json.loads(content)
            # 校验字段
            required = ["start_date", "end_date", "sector_name", "indicator"]
            for k in required:
                if k not in params:
                    raise ValueError(f"缺少字段: {k}")
            return params
        except Exception as e:
            raise RuntimeError(f"LLM 解析失败: {e}")

    def fetch_data(self, params):
        """
        根据参数获取板块历史资金流向数据
        返回 DataFrame，列：date, net_inflow, change_pct, amount
        """
        sector_name = params["sector_name"]
        start_str = params["start_date"]
        end_str = params["end_date"]

        start = safe_str_to_date(start_str)
        end = safe_str_to_date(end_str)
        if start > end:
            start, end = end, start

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if end > today:
            end = today
        if start > today:
            raise ValueError("起始日期不能晚于今天")

        # 生成日期列表（自然日）
        date_list = pd.date_range(start, end, freq='D').tolist()
        results = []

        print(f"开始获取 {sector_name} 板块历史资金流向，共 {len(date_list)} 个自然日...")
        for i, dt in enumerate(date_list):
            d_str = dt.strftime("%Y-%m-%d")
            # 跳过未来日期
            if dt > today:
                continue

            # 每10天打印进度
            if i % 10 == 0:
                print(f"进度: {i+1}/{len(date_list)}")

            try:
                # 获取行业板块排行（历史）
                data = lk.sector_ranking_kph(
                    date=d_str,
                    zs_type=lk.SECTOR_INDUSTRY,
                    fetch_all=True
                )
                if not data:
                    continue

                # 查找目标板块（模糊匹配）
                found = None
                for item in data:
                    if sector_name.lower() in item.get('plate_name', '').lower():
                        found = item
                        break
                if found:
                    results.append({
                        'date': d_str,
                        'net_inflow': found.get('net_inflow', 0),
                        'change_pct': found.get('change_pct', 0),
                        'amount': found.get('amount', 0),
                        'plate_name': found.get('plate_name', '')
                    })
            except Exception as e:
                # 非交易日或接口错误，跳过
                continue

        if not results:
            raise RuntimeError(f"未找到板块 '{sector_name}' 的任何历史数据，请检查板块名称。")

        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        print(f"成功获取 {len(df)} 个交易日数据")
        return df

    def save_csv_and_plot(self, df, sector_name, start_date, end_date):
        """保存 CSV 和绘制资金趋势图"""
        # 文件名
        safe_sector = re.sub(r'[^\w]', '_', sector_name)
        file_prefix = f"{safe_sector}_{start_date}_{end_date}"
        csv_path = os.path.join(DATA_DIR, f"{file_prefix}_资金流向.csv")
        png_path = os.path.join(DATA_DIR, f"{file_prefix}_资金趋势.png")

        # 保存 CSV
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"CSV 已保存: {csv_path}")

        # 绘制折线图
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['net_inflow'], marker='o', linestyle='-', linewidth=2, markersize=4)
        plt.title(f"{sector_name} 板块主力净流入趋势\n({start_date} - {end_date})", fontsize=14)
        plt.xlabel("日期", fontsize=12)
        plt.ylabel("主力净流入 (元)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)

        # 优化 x 轴刻度密度
        if len(df) > 30:
            step = max(1, len(df) // 30)
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))

        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"趋势图已保存: {png_path}")

    def run(self):
        """命令行交互主循环"""
        print("=== A股数据助手 (DeepSeek + levistock) ===")
        print("输入自然语言指令，例如：'20210101-20260101半导体资金流向'")
        print("输入 'quit' 或 'exit' 退出")

        while True:
            try:
                user_input = input("\n> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ('quit', 'exit'):
                    print("再见！")
                    break

                # 1. 解析
                params = self.parse_request(user_input)
                print(f"解析结果: {params}")

                # 2. 获取数据
                df = self.fetch_data(params)

                # 3. 保存
                self.save_csv_and_plot(
                    df,
                    params['sector_name'],
                    params['start_date'],
                    params['end_date']
                )
                print("✅ 任务完成")

            except KeyboardInterrupt:
                print("\n退出")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")

# ================== 主入口 ==================
if __name__ == "__main__":
    # 检查 API Key
    if DEEPSEEK_API_KEY == "your-api-key-here":
        print("请先设置环境变量 DEEPSEEK_API_KEY，或在代码中填入你的 API Key")
        # 可以在这里让用户输入
        key = input("请输入 DeepSeek API Key: ").strip()
        if key:
            DEEPSEEK_API_KEY = key
        else:
            print("未提供 API Key，程序退出")
            exit(1)

    agent = StockAgent(api_key='sk-cb7ba47a299d4ebc8e14bafe6b41f33c')
    agent.run()