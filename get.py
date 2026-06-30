# agent.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from datetime import datetime, timedelta
from zzshare.client import DataApi
import time
import re
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class StockDataAgent:
    """股票数据获取Agent，使用DeepSeek LLM进行自然语言理解"""
    
    def __init__(self):
        """初始化Agent"""
        # 直接初始化，不需要token
        self.api = DataApi()
        self.data_dir = "stock_data"
        self._ensure_directories()
        
        # 初始化DeepSeek客户端
        self._init_deepseek()
        
        # 命令模式标记
        self.running = True
        
    def _init_deepseek(self):
        """初始化DeepSeek客户端"""
        try:
            from openai import OpenAI
            
            # 从环境变量读取API密钥
            api_key = 'sk-cb7ba47a299d4ebc8e14bafe6b41f33c'
            
            if not api_key:
                print("⚠️ 未设置DEEPSEEK_API_KEY环境变量")
                print("请设置: export DEEPSEEK_API_KEY='your-api-key'")
                print("或者将API密钥写入 .env 文件")
                print("💡 提示: 可以在 https://platform.deepseek.com/ 获取API密钥")
                self.client = None
                return
                
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            print("✅ DeepSeek客户端初始化成功")
            
        except ImportError:
            print("⚠️ 未安装openai库，请运行: pip install openai")
            print("💡 不使用DeepSeek也可以运行，将使用规则匹配模式")
            self.client = None
        except Exception as e:
            print(f"⚠️ DeepSeek初始化失败: {e}")
            print("💡 将使用规则匹配模式（无需API密钥）")
            self.client = None
            
    def _ensure_directories(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def _parse_natural_language(self, query: str) -> Dict:
        """
        使用DeepSeek解析自然语言查询
        
        Args:
            query: 用户输入的自然语言查询
            
        Returns:
            解析后的参数字典
        """
        # 如果DeepSeek不可用，使用规则匹配
        if self.client is None:
            return self._parse_with_rules(query)
            
        try:
            prompt = f"""请解析以下股票数据查询请求，提取关键参数，并以JSON格式返回。

查询: {query}

请提取以下参数（如果存在）：
1. stock_codes: 股票代码列表（如 ["600330", "002428"]），如果提到股票名称则返回对应股票代码
2. start_date: 开始日期（格式 YYYYMMDD），如果提到"最近"、"近期"则留空
3. end_date: 结束日期（格式 YYYYMMDD），如果提到"最近"、"近期"则留空
4. action: 操作类型（fetch-获取数据, analyze-分析, visualize-可视化, all-全部）

输出格式（JSON）：
{{
    "stock_codes": [],
    "start_date": "",
    "end_date": "",
    "action": "all"
}}

只返回JSON，不要其他内容。"""

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个股票数据查询参数解析专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            # 提取JSON
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                params = json.loads(json_match.group())
                return params
            else:
                return self._parse_with_rules(query)
                
        except Exception as e:
            print(f"⚠️ DeepSeek解析失败: {e}，使用规则解析")
            return self._parse_with_rules(query)
            
    def _parse_with_rules(self, query: str) -> Dict:
        """
        使用规则匹配解析查询（备用方案）
        
        Args:
            query: 用户查询字符串
            
        Returns:
            解析后的参数字典
        """
        params = {
            "stock_codes": [],
            "start_date": "",
            "end_date": "",
            "action": "all"
        }
        
        # 提取股票代码（6位数字）
        codes = re.findall(r'\b\d{6}\b', query)
        if codes:
            params["stock_codes"] = codes
            
        # 提取日期（YYYYMMDD格式）
        date_pattern = re.compile(r'\b(19|20)\d{6}\b')
        all_dates = date_pattern.findall(query)
        if all_dates:
            if len(all_dates) >= 2:
                params["start_date"] = all_dates[0]
                params["end_date"] = all_dates[1]
            elif len(all_dates) == 1:
                params["end_date"] = all_dates[0]
                
        # 检查是否提到"最近"
        if "最近" in query or "近期" in query:
            params["end_date"] = datetime.now().strftime("%Y%m%d")
            # 默认最近一年
            if not params["start_date"]:
                end_dt = datetime.now()
                start_dt = end_dt - timedelta(days=365)
                params["start_date"] = start_dt.strftime("%Y%m%d")
                
        # 检查关键词
        if "分析" in query or "统计" in query:
            params["action"] = "analyze"
        elif "图" in query or "走势" in query or "可视化" in query:
            params["action"] = "visualize"
        elif "全部" in query or "所有" in query:
            params["action"] = "all"
            
        return params
        
    def fetch_stock_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取单只股票数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            股票数据DataFrame
        """
        try:
            # 格式化股票代码
            if len(stock_code) == 6:
                # 判断交易所
                if stock_code.startswith('6'):
                    ts_code = f"{stock_code}.SH"
                elif stock_code.startswith('0') or stock_code.startswith('3'):
                    ts_code = f"{stock_code}.SZ"
                else:
                    ts_code = f"{stock_code}.SH"  # 默认上海
            else:
                ts_code = stock_code
                
            print(f"📊 获取 {ts_code} 数据...")
            
            # 计算需要获取的数据量
            start = datetime.strptime(start_date, "%Y%m%d")
            end = datetime.strptime(end_date, "%Y%m%d")
            days = (end - start).days
            
            # 如果数据量超过500条，分页获取
            if days > 500:
                all_data = []
                offset = 0
                limit = 500
                max_pages = 20  # 最多获取20页，防止死循环
                page = 0
                
                while page < max_pages:
                    df = self.api.daily(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date,
                        offset=offset,
                        limit=limit,
                        fields='all'
                    )
                    
                    if df is None or df.empty:
                        break
                        
                    all_data.append(df)
                    offset += limit
                    page += 1
                    
                    if len(df) < limit:
                        break
                        
                    time.sleep(0.3)  # 避免请求过快
                    
                if all_data:
                    df = pd.concat(all_data, ignore_index=True)
                else:
                    return None
            else:
                df = self.api.daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields='all'
                )
                
            if df is None or df.empty:
                print(f"⚠️ 未获取到 {ts_code} 的数据")
                return None
                
            print(f"✅ 获取 {ts_code} 数据成功，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            print(f"❌ 获取 {stock_code} 数据失败: {e}")
            return None
            
    def save_stock_data(self, stock_code: str, df: pd.DataFrame, start_date: str, end_date: str):
        """
        保存股票数据到文件夹
        """
        # 创建股票专属文件夹
        stock_dir = os.path.join(self.data_dir, stock_code)
        if not os.path.exists(stock_dir):
            os.makedirs(stock_dir)
        
        # 确保 trade_date 是数值类型并过滤未来日期
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_numeric(df['trade_date'])
            today_int = int(datetime.now().strftime('%Y%m%d'))
            df = df[df['trade_date'] <= today_int]
        
        # 按交易日期升序排序（从旧到新）
        if 'trade_date' in df.columns and not df.empty:
            df = df.sort_values('trade_date').reset_index(drop=True)
            print(f"📅 数据已按日期升序排列 (从 {df['trade_date'].iloc[0]} 到 {df['trade_date'].iloc[-1]})")
        
        # 保存CSV
        csv_file = os.path.join(stock_dir, f"{stock_code}_{start_date}_{end_date}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"💾 数据已保存: {csv_file}")
        
        # 保存股票基本信息
        # 保存股票基本信息
        info_file = os.path.join(stock_dir, f"{stock_code}_info.json")
        info = {
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date,
            "data_count": len(df),
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date_range": {
                "first_date": int(df['trade_date'].min()) if not df.empty else None,  # 加 int()
                "last_date": int(df['trade_date'].max()) if not df.empty else None    # 加 int()
            },
            "price_range": {
                "high": float(df['high'].max()) if not df.empty and 'high' in df.columns else None,
                "low": float(df['low'].min()) if not df.empty and 'low' in df.columns else None
            },
            "total_volume": float(df['vol'].sum()) if not df.empty and 'vol' in df.columns else None
        }
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"💾 基本信息已保存: {info_file}")
        
        # 生成股价走势图
        self.plot_stock_chart(stock_code, df, start_date, end_date)
        
    def plot_stock_chart(self, stock_code: str, df: pd.DataFrame, start_date: str, end_date: str):
        """
        生成股价走势图
        """
        print(f"📊 数据列: {df.columns.tolist()}")
        if 'vol' in df.columns:
            print(f"📊 vol 数据类型: {df['vol'].dtype}")
            print(f"📊 vol 前5行: {df['vol'].head().tolist()}")
            print(f"📊 vol 最大值: {df['vol'].max()}, 最小值: {df['vol'].min()}")
            print(f"📊 vol 是否全为0: {(df['vol'] == 0).all()}")
            print(f"📊 vol 是否有空值: {df['vol'].isna().any()}")
            
        try:
            if df.empty:
                print(f"⚠️ {stock_code} 数据为空，无法生成图表")
                return
                
            # 确保日期列存在
            if 'trade_date' not in df.columns:
                print(f"⚠️ {stock_code} 数据缺少日期列")
                return
                
            # 排序
            df_sorted = df.sort_values('trade_date').reset_index(drop=True)
            
            # 将 trade_date 转换为日期字符串
            df_sorted['trade_date_str'] = df_sorted['trade_date'].astype(str).apply(
                lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}"
            )
            
            # ===== 诊断成交量数据 =====
            vol_col = None
            for col in ['vol', 'volume', 'VOL', 'Volume']:
                if col in df_sorted.columns:
                    vol_col = col
                    break
            
            if vol_col is None:
                print(f"⚠️ 未找到成交量列，可用列: {df_sorted.columns.tolist()}")
            else:
                # 确保成交量是数值类型
                df_sorted[vol_col] = pd.to_numeric(df_sorted[vol_col], errors='coerce')
                # 填充NaN为0
                df_sorted[vol_col] = df_sorted[vol_col].fillna(0)
                print(f"📊 使用成交量列: {vol_col}, 最大值: {df_sorted[vol_col].max():.0f}")
            
            # 创建索引用于绘图
            x_indices = range(len(df_sorted))
            
            # 创建图表
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), 
                                    gridspec_kw={'height_ratios': [3, 1]})
            
            # 主图：价格走势
            ax1 = axes[0]
            if 'close' in df_sorted.columns:
                ax1.plot(x_indices, df_sorted['close'], 
                        label='收盘价', linewidth=2, color='#1f77b4')
                
            if 'high' in df_sorted.columns and 'low' in df_sorted.columns:
                ax1.fill_between(x_indices, 
                                df_sorted['low'], df_sorted['high'],
                                alpha=0.2, color='#1f77b4', label='价格区间')
                
            ax1.set_title(f'{stock_code} 股价走势 ({start_date} - {end_date})', 
                        fontsize=16, fontweight='bold')
            ax1.set_ylabel('价格 (元)')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 子图：成交量
            ax2 = axes[1]
            if vol_col is not None:
                # 检查成交量是否有有效数据
                vol_data = df_sorted[vol_col]
                if vol_data.sum() > 0:
                    # 成交量可能太大，转换为"万手"或"亿股"显示
                    max_vol = vol_data.max()
                    if max_vol > 100000000:  # 超过1亿
                        display_vol = vol_data / 100000000
                        unit = '亿股'
                    elif max_vol > 10000:  # 超过1万
                        display_vol = vol_data / 10000
                        unit = '万股'
                    else:
                        display_vol = vol_data
                        unit = '股'
                    
                    ax2.bar(x_indices, display_vol, 
                        color='#2ca02c', alpha=0.7, label=f'成交量({unit})', width=0.8)
                    ax2.set_ylabel(f'成交量 ({unit})')
                else:
                    ax2.text(0.5, 0.5, '⚠️ 成交量数据全为0', 
                            transform=ax2.transAxes, ha='center', va='center',
                            fontsize=14, color='red')
                    print(f"⚠️ {stock_code} 成交量全为0")
            else:
                ax2.text(0.5, 0.5, f'⚠️ 无成交量列: {df_sorted.columns.tolist()}', 
                        transform=ax2.transAxes, ha='center', va='center',
                        fontsize=10, color='red')
            
            ax2.set_xlabel('日期')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # 统一设置x轴刻度标签
            total_points = len(df_sorted)
            if total_points > 0:
                if total_points > 50:
                    step = max(1, total_points // 30)
                    display_indices = list(range(0, total_points, step))
                else:
                    display_indices = list(range(total_points))
                
                display_labels = [df_sorted['trade_date_str'].iloc[i] for i in display_indices]
                
                ax1.set_xticks(display_indices)
                ax1.set_xticklabels(display_labels, rotation=45, ha='right')
                
                ax2.set_xticks(display_indices)
                ax2.set_xticklabels(display_labels, rotation=45, ha='right')
            
            plt.tight_layout()
            
            # 保存图片
            stock_dir = os.path.join(self.data_dir, stock_code)
            img_file = os.path.join(stock_dir, f"{stock_code}_{start_date}_{end_date}.png")
            plt.savefig(img_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"📈 走势图已保存: {img_file}")
            
        except Exception as e:
            print(f"❌ 生成 {stock_code} 图表失败: {e}")
            import traceback
            traceback.print_exc()
            
    def process_request(self, query: str):
        """
        处理用户请求
        
        Args:
            query: 用户输入的自然语言查询
        """
        print("\n" + "="*60)
        print(f"📝 处理请求: {query}")
        print("="*60)
        
        # 解析查询
        params = self._parse_natural_language(query)
        
        stock_codes = params.get("stock_codes", [])
        start_date = params.get("start_date", "")
        end_date = params.get("end_date", "")
        action = params.get("action", "all")
        
        # 如果未指定日期，默认最近一年
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        if not start_date:
            end_dt = datetime.strptime(end_date, "%Y%m%d")
            start_dt = end_dt - timedelta(days=365)
            start_date = start_dt.strftime("%Y%m%d")
            
        print(f"📊 参数解析结果:")
        print(f"  股票代码: {stock_codes if stock_codes else '未指定'}")
        print(f"  开始日期: {start_date}")
        print(f"  结束日期: {end_date}")
        print(f"  操作类型: {action}")
        
        # 如果未指定股票代码，提示用户
        if not stock_codes:
            print("⚠️ 未识别到股票代码，请指定股票代码（6位数字）")
            print("💡 例如: 600330, 000001, 002428")
            return
            
        # 处理每个股票
        for code in stock_codes:
            print(f"\n{'='*50}")
            print(f"📈 处理股票: {code}")
            print(f"{'='*50}")
            
            # 获取数据
            df = self.fetch_stock_data(code, start_date, end_date)
            
            if df is not None and not df.empty:
                # 保存数据和图表
                self.save_stock_data(code, df, start_date, end_date)
                
                # 如果有分析需求
                if action in ["analyze", "all"]:
                    self.analyze_stock(code, df)
                    
            time.sleep(0.5)  # 避免请求过快
            
        print(f"\n✅ 所有请求处理完成！")
        print(f"📁 数据保存在: {os.path.abspath(self.data_dir)}")
        
    def analyze_stock(self, stock_code: str, df: pd.DataFrame):
        """
        分析股票数据（可选功能）
        
        Args:
            stock_code: 股票代码
            df: 股票数据
        """
        try:
            if df.empty:
                return
                
            print(f"\n📊 {stock_code} 数据分析:")
            print("-" * 40)
            
            # 基础统计
            if 'close' in df.columns:
                close = df['close']
                print(f"  最新收盘价: {close.iloc[-1]:.2f}")
                print(f"  最高价: {close.max():.2f}")
                print(f"  最低价: {close.min():.2f}")
                print(f"  平均价: {close.mean():.2f}")
                
                # 涨跌幅
                if len(close) > 1:
                    change = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
                    print(f"  区间涨跌幅: {change:.2f}%")
                    
            if 'vol' in df.columns:
                vol = df['vol']
                print(f"  总成交量: {vol.sum():.2e}")
                print(f"  日均成交量: {vol.mean():.2e}")
                
            # 保存分析结果
            stock_dir = os.path.join(self.data_dir, stock_code)
            analysis_file = os.path.join(stock_dir, f"{stock_code}_analysis.json")
            
            analysis = {
                "stock_code": stock_code,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "statistics": {
                    "latest_close": float(df['close'].iloc[-1]) if 'close' in df.columns else None,
                    "high": float(df['close'].max()) if 'close' in df.columns else None,
                    "low": float(df['close'].min()) if 'close' in df.columns else None,
                    "average": float(df['close'].mean()) if 'close' in df.columns else None,
                    "total_volume": float(df['vol'].sum()) if 'vol' in df.columns else None,
                    "average_volume": float(df['vol'].mean()) if 'vol' in df.columns else None
                }
            }
            
            if 'close' in df.columns and len(df['close']) > 1:
                change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
                analysis["statistics"]["change_percent"] = float(change)
                
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
                
            print(f"💾 分析结果已保存: {analysis_file}")
            
        except Exception as e:
            print(f"❌ 分析 {stock_code} 失败: {e}")
            
    def run_interactive(self):
        """运行交互式命令行界面"""
        print("\n" + "="*60)
        print("🚀 股票数据智能Agent (基于DeepSeek)")
        print("="*60)
        print("\n💡 使用说明:")
        print("  1. 输入自然语言查询，例如:")
        print("     '获取 600330,002428 的股票数据'")
        print("     '帮我分析 600330 20230101 到 20231231 的数据'")
        print("     '给我 600330,002428,002222 所有股票 2020到2024的数据'")
        print("     '获取 600330 最近一年的数据'")
        print("  2. 输入 'exit' 或 'quit' 退出")
        print("  3. 输入 'help' 显示帮助")
        print("\n" + "="*60)
        
        # 检查DeepSeek状态
        if self.client is None:
            print("\n⚠️ DeepSeek未配置，将使用规则解析模式")
            print("   💡 如需使用DeepSeek，请设置环境变量:")
            print("   export DEEPSEEK_API_KEY='your-api-key'")
            print("   或在 .env 文件中配置")
            print("   🔑 获取密钥: https://platform.deepseek.com/")
        else:
            print("\n✅ DeepSeek已就绪，支持自然语言理解")
            
        print("\n🎯 zzshare已就绪（无需token）")
        print("\n等待您的输入...\n")
        
        while self.running:
            try:
                query = input("📝 您: ").strip()
                
                if not query:
                    continue
                    
                if query.lower() in ['exit', 'quit', 'q']:
                    print("👋 再见！")
                    break
                    
                if query.lower() == 'help':
                    self._show_help()
                    continue
                    
                # 处理请求
                self.process_request(query)
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 处理请求时发生错误: {e}")
                
    def _show_help(self):
        """显示帮助信息"""
        print("\n" + "="*60)
        print("📖 帮助信息")
        print("="*60)
        print("\n支持的自然语言示例:")
        print("  • '获取 600330,002428 的股票数据'")
        print("  • '帮我分析 600330 20230101 到 20231231 的数据'")
        print("  • '获取 600330 的所有数据并生成图表'")
        print("  • '给我 600330,002428,002222 所有股票 2020到2024的数据'")
        print("  • '获取 600330 最近一年的数据'")
        print("  • '获取 600330 2024年的数据'")
        print("\n支持的格式:")
        print("  • 股票代码: 6位数字 (如 600330)")
        print("  • 日期: YYYYMMDD格式 (如 20230101)")
        print("  • 日期范围: 可以用'到'、'-'、'至'等连接")
        print("\n输出文件:")
        print("  • CSV文件: 包含所有交易数据")
        print("  • PNG图片: 股价走势图和成交量")
        print("  • JSON文件: 股票基本信息和统计分析")
        print("\n命令:")
        print("  • help  - 显示此帮助")
        print("  • exit  - 退出程序")
        print("="*60 + "\n")


def main():
    """主函数"""
    # 创建Agent（不需要token）
    agent = StockDataAgent()
    
    # 运行交互式界面
    agent.run_interactive()


if __name__ == "__main__":
    main()