# agent.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
from zzshare.client import DataApi
import time
import re
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class StockDataAgent:
    """股票数据智能Agent - 基于zzshare库，动态适配可用接口"""

    def __init__(self):
        self.api = DataApi()
        self.data_dir = "stock_data"
        self._ensure_directories()
        self._init_deepseek()
        self.running = True

        # 候选接口列表（依据文档整理，按类别）
        candidate_methods = [
            # 基础行情
            'daily', 'stk_mins', 'rt_k',
            # 基础数据
            'stock_basic', 'trade_days', 'stock_info',
            # 涨停复盘
            'uplimit_hot', 'uplimit_stocks', 'stock_uplimit_reason',
            'review_uplimit_reason', 'review_uplimit_hot_step', 'review_uplimit_reason_open',
            # 龙虎榜
            'lhb_list', 'lhb_detail', 'lhb_stock_history', 'lhb_trader_history',
            # 情绪热度
            'market_sentiment', 'market_hot_sentiment', 'market_style',
            'open_sentiment_data', 'sentiment_market_hot_day',
            'sentiment_trend', 'sentiment_trend_range',
            'updown_distribution', 'uplimit_trend', 'sentiment_hot_day',
            'sentiment_bull_data', 'ths_hot_top', 'stock_ths_hot',
            # 板块分析
            'plates_list', 'plates_rank', 'market_plate_stocks',
            'plates_trend', 'plates_stocks', 'plate_kline',
            # 题材库
            'topic_table_list', 'topic_table_detail', 'topic_table_stocks', 'topic_kline',
            # 其他
            'uplimit_market_value', 'sentiment_market_top_n',
            'movement_alerts', 'zdjk_get',
            # 资金流向（可能需要token）
            'stock_moneyflow', 'market_mf',
        ]

        self.api_mapping = {}
        print("\n🔍 正在检测可用的API接口...")
        for name in candidate_methods:
            if hasattr(self.api, name):
                self.api_mapping[name] = {
                    'method': getattr(self.api, name),
                    'desc': self._get_desc(name)
                }
                print(f"  ✅ {name}")
            else:
                # 不输出每个缺失的，避免刷屏，但可以记录
                pass
        print(f"✅ 共加载 {len(self.api_mapping)} 个接口\n")

        # 意图关键词映射（仅映射到已存在的接口）
        self.intent_patterns = {}
        for api_name in self.api_mapping.keys():
            # 为每个接口设置关键词（从接口名推断）
            keywords = self._generate_keywords(api_name)
            if keywords:
                self.intent_patterns[api_name] = keywords

    def _get_desc(self, name):
        """获取接口简短描述"""
        desc_map = {
            'daily': '历史日线',
            'stk_mins': '分钟K线',
            'rt_k': '实时快照',
            'stock_basic': '股票基础信息',
            'trade_days': '交易日历',
            'stock_info': '个股资料',
            'uplimit_hot': '涨停热门板块',
            'uplimit_stocks': '涨停股票列表',
            'stock_uplimit_reason': '涨停原因',
            'review_uplimit_reason': '涨停原因复盘',
            'review_uplimit_hot_step': '涨停热门',
            'review_uplimit_reason_open': '涨停原因',
            'lhb_list': '龙虎榜列表',
            'lhb_detail': '龙虎榜详情',
            'lhb_stock_history': '个股龙虎榜历史',
            'lhb_trader_history': '席位交易历史',
            'market_sentiment': '市场情绪K线',
            'market_hot_sentiment': '热门情绪K线',
            'market_style': '市场风格择时',
            'open_sentiment_data': '情绪数据',
            'sentiment_market_hot_day': '当日市场热度',
            'sentiment_trend': '情绪趋势',
            'sentiment_trend_range': '情绪趋势区间',
            'updown_distribution': '涨跌分布',
            'uplimit_trend': '涨停趋势',
            'sentiment_hot_day': '日度市场热度',
            'sentiment_bull_data': '牛熊情绪',
            'ths_hot_top': '同花顺热度排行',
            'stock_ths_hot': '个股同花顺热度',
            'plates_list': '板块列表',
            'plates_rank': '板块排名',
            'market_plate_stocks': '板块成分股',
            'plates_trend': '板块趋势',
            'plates_stocks': '板块成分股',
            'plate_kline': '板块K线',
            'topic_table_list': '题材库表格列表',
            'topic_table_detail': '题材库表格详情',
            'topic_table_stocks': '题材下个股列表',
            'topic_kline': '题材合成指数K线',
            'uplimit_market_value': '涨停市值统计',
            'sentiment_market_top_n': '市场TopN情绪',
            'movement_alerts': '异动数据',
            'zdjk_get': '监控数据',
            'stock_moneyflow': '个股资金流向',
            'market_mf': '市场资金流向分钟',
        }
        return desc_map.get(name, name)

    def _generate_keywords(self, api_name):
        """根据接口名生成中文关键词列表"""
        base_map = {
            'daily': ['日线', '日K', '历史行情', '股价', '走势', 'K线'],
            'stk_mins': ['分钟', '分时', '日内', '1min', '5min', '分钟线'],
            'rt_k': ['实时', '快照', '当前', '最新', '现价', '实时行情'],
            'stock_basic': ['股票列表', '所有股票', '全部股票', '股票代码', '上市公司'],
            'trade_days': ['交易日', '开盘日', '交易日期', '休市'],
            'stock_info': ['个股资料', '公司信息', '股票信息'],
            'uplimit_hot': ['涨停板块', '热门板块', '涨停热点', '涨停概念'],
            'uplimit_stocks': ['涨停', '跌停', '封板', '连板', '涨停股'],
            'stock_uplimit_reason': ['涨停原因', '为什么涨停', '封板原因', '涨停理由'],
            'review_uplimit_reason': ['涨停复盘', '原因复盘'],
            'review_uplimit_hot_step': ['涨停热门', '热门涨停'],
            'review_uplimit_reason_open': ['涨停原因', '复盘原因'],
            'lhb_list': ['龙虎榜', '席位', '游资', '机构买入', '上榜'],
            'lhb_detail': ['龙虎榜详情', '龙虎榜明细', '上榜详情'],
            'lhb_stock_history': ['个股龙虎榜', '龙虎榜历史'],
            'lhb_trader_history': ['席位交易', '营业部'],
            'market_sentiment': ['情绪', '市场热度', '情绪指标', '牛熊', '市场情绪'],
            'market_hot_sentiment': ['热门情绪'],
            'market_style': ['风格择时', '市场风格'],
            'open_sentiment_data': ['情绪数据'],
            'sentiment_market_hot_day': ['当日热度', '市场热度'],
            'sentiment_trend': ['情绪趋势'],
            'sentiment_trend_range': ['情绪区间'],
            'updown_distribution': ['涨跌分布'],
            'uplimit_trend': ['涨停趋势'],
            'sentiment_hot_day': ['日度热度'],
            'sentiment_bull_data': ['牛熊情绪'],
            'ths_hot_top': ['热度', '热搜', '热门股票', '人气', '热榜'],
            'stock_ths_hot': ['个股热度'],
            'plates_list': ['板块列表', '所有板块', '行业板块', '概念板块'],
            'plates_rank': ['板块排名', '板块涨幅', '板块涨跌', '板块表现'],
            'market_plate_stocks': ['成分股', '板块个股', '板块股票', '板块成分'],
            'plates_trend': ['板块趋势'],
            'plates_stocks': ['板块成分股'],
            'plate_kline': ['板块K线', '板块指数', '板块走势'],
            'topic_table_list': ['题材', '概念', '题材库', '题材列表'],
            'topic_table_detail': ['题材详情'],
            'topic_table_stocks': ['题材个股'],
            'topic_kline': ['题材K线'],
            'uplimit_market_value': ['涨停市值'],
            'sentiment_market_top_n': ['TopN情绪'],
            'movement_alerts': ['异动', '异常波动', '异动股'],
            'zdjk_get': ['监控数据'],
            'stock_moneyflow': ['资金流向', '主力资金', '资金流入', '净流入'],
            'market_mf': ['市场资金', '资金流向分钟'],
        }
        return base_map.get(api_name, [])

    def _init_deepseek(self):
        """初始化DeepSeek客户端"""
        try:
            from openai import OpenAI
            api_key = 'sk-cb7ba47a299d4ebc8e14bafe6b41f33c'
            if not api_key:
                print("⚠️ 未设置DEEPSEEK_API_KEY环境变量，将使用规则解析")
                self.client = None
                return
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            print("✅ DeepSeek客户端初始化成功")
        except Exception as e:
            print(f"⚠️ DeepSeek初始化失败: {e}，使用规则解析")
            self.client = None

    def _ensure_directories(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _parse_natural_language(self, query: str) -> Dict:
        """使用DeepSeek或规则解析查询"""
        if self.client is not None:
            try:
                return self._parse_with_deepseek(query)
            except Exception as e:
                print(f"⚠️ DeepSeek解析失败: {e}，回退到规则解析")
        return self._parse_with_rules(query)

    def _parse_with_deepseek(self, query: str) -> Dict:
        """DeepSeek解析"""
        # 构建可用接口摘要
        api_list = "\n".join([f"- {name}: {desc}" for name, desc in 
                             [(n, self.api_mapping[n]['desc']) for n in self.api_mapping.keys()]])
        prompt = f"""请解析以下股票数据查询请求，提取关键参数，选择合适的API接口。

查询: {query}

可用接口（仅限以下）:
{api_list}

请以JSON格式返回：
{{
    "api_name": "选择的API名称（必须从上述列表中选择）",
    "params": {{
        "ts_code": "股票代码(如600000.SH或600000)",
        "start_date": "开始日期(YYYYMMDD)",
        "end_date": "结束日期(YYYYMMDD)",
        "trade_date": "交易日期(YYYYMMDD)",
        "date1": "日期1(YYYYMMDD)",
        "date2": "日期2(YYYYMMDD)",
        "stock_code": "股票代码",
        "stock_id": "股票代码",
        "exchange": "交易所(SSE/SZSE/ALL/KSH/BSE/GEM)",
        "freq": "频率(1min/5min/15min/30min/60min)",
        "plate_type": "板块类型(17题材/15概念/14行业)",
        "plate_code": "板块代码",
        "b_code": "板块代码",
        "limit": 返回数量,
        "offset": 偏移量,
        "fields": "返回字段",
        "adj": "复权类型(qfq/hfq)",
        "list_status": "上市状态(L/D/P)"
    }},
    "data_type": "数据类型(daily/mins/realtime/basic/limit/sentiment/plate/topic/alert/moneyflow)",
    "description": "查询描述"
}}
只返回JSON，不要其他内容。"""

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个股票数据查询专家，精通zzshare库的所有接口。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        result_text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            params = json.loads(json_match.group())
            # 确保api_name存在
            if params.get('api_name') not in self.api_mapping:
                # 尝试寻找最接近的
                for key in self.api_mapping.keys():
                    if key in params.get('api_name', ''):
                        params['api_name'] = key
                        break
                else:
                    # 默认daily
                    params['api_name'] = 'daily'
            return params
        else:
            return self._parse_with_rules(query)

    def _parse_with_rules(self, query: str) -> Dict:
        """规则解析"""
        result = {
            "api_name": "daily",
            "params": {},
            "data_type": "daily",
            "description": query
        }

        # 根据关键词匹配接口
        for api_name, keywords in self.intent_patterns.items():
            if any(kw in query for kw in keywords):
                result['api_name'] = api_name
                break

        # 提取股票代码
        codes = re.findall(r'\b\d{6}\b', query)
        if codes:
            code = codes[0]
            if code.startswith('6'):
                ts_code = f"{code}.SH"
            elif code.startswith('0') or code.startswith('3'):
                ts_code = f"{code}.SZ"
            elif code.startswith('8') or code.startswith('4'):
                ts_code = f"{code}.BJ"
            else:
                ts_code = f"{code}.SH"
            result['params']['ts_code'] = ts_code
            result['params']['stock_code'] = code
            result['params']['stock_id'] = code

        # 提取日期
        year_pattern = re.compile(r'(\d{4})(?:到|至|-)(\d{4,8})')
        year_match = year_pattern.search(query)
        if year_match:
            start = year_match.group(1)
            end = year_match.group(2)
            if len(end) == 4:
                result['params']['start_date'] = start + "0101"
                result['params']['end_date'] = end + "1231"
                result['params']['date1'] = start + "0101"
                result['params']['date2'] = end + "1231"
            else:
                result['params']['start_date'] = start + "0101"
                result['params']['end_date'] = end
                result['params']['date1'] = start + "0101"
                result['params']['date2'] = end
        else:
            date_pattern = re.compile(r'\b(19|20)\d{6}\b')
            dates = date_pattern.findall(query)
            if dates:
                if len(dates) >= 2:
                    result['params']['start_date'] = dates[0]
                    result['params']['end_date'] = dates[1]
                    result['params']['date1'] = dates[0]
                    result['params']['date2'] = dates[1]
                else:
                    result['params']['trade_date'] = dates[0]
                    result['params']['date1'] = dates[0]

        if not result['params'].get('start_date'):
            year_pattern = re.compile(r'(\d{4})年')
            years = year_pattern.findall(query)
            if years:
                if len(years) >= 2:
                    result['params']['start_date'] = years[0] + "0101"
                    result['params']['end_date'] = years[1] + "1231"
                    result['params']['date1'] = years[0] + "0101"
                    result['params']['date2'] = years[1] + "1231"
                else:
                    result['params']['start_date'] = years[0] + "0101"
                    result['params']['end_date'] = years[0] + "1231"
                    result['params']['date1'] = years[0] + "0101"
                    result['params']['date2'] = years[0] + "1231"

        # 交易所
        if "沪市" in query or "上证" in query:
            result['params']['exchange'] = "SSE"
        elif "深市" in query or "深证" in query:
            result['params']['exchange'] = "SZSE"
        elif "科创板" in query:
            result['params']['exchange'] = "KSH"
        elif "创业板" in query:
            result['params']['exchange'] = "GEM"
        elif "北交所" in query:
            result['params']['exchange'] = "BSE"
        elif "全部" in query or "所有" in query:
            result['params']['exchange'] = "ALL"

        # 频率
        if "1min" in query or "1分钟" in query:
            result['params']['freq'] = "1min"
        elif "5min" in query or "5分钟" in query:
            result['params']['freq'] = "5min"
        elif "15min" in query or "15分钟" in query:
            result['params']['freq'] = "15min"
        elif "30min" in query or "30分钟" in query:
            result['params']['freq'] = "30min"
        elif "60min" in query or "60分钟" in query:
            result['params']['freq'] = "60min"

        # 复权
        if "前复权" in query:
            result['params']['adj'] = "qfq"
        elif "后复权" in query:
            result['params']['adj'] = "hfq"

        # 板块类型
        if "题材" in query:
            result['params']['plate_type'] = 17
        elif "概念" in query:
            result['params']['plate_type'] = 15
        elif "行业" in query:
            result['params']['plate_type'] = 14

        # limit
        limit_match = re.search(r'(\d+)\s*(?:条|个|只)', query)
        if limit_match:
            result['params']['limit'] = int(limit_match.group(1))

        # data_type推断
        if result['api_name'] in ['daily', 'stk_mins', 'rt_k']:
            result['data_type'] = 'daily' if result['api_name'] == 'daily' else 'mins'
        elif result['api_name'] in ['uplimit_hot', 'uplimit_stocks']:
            result['data_type'] = 'limit'
        elif 'sentiment' in result['api_name']:
            result['data_type'] = 'sentiment'
        elif 'plate' in result['api_name']:
            result['data_type'] = 'plate'
        elif 'topic' in result['api_name']:
            result['data_type'] = 'topic'
        elif 'alert' in result['api_name']:
            result['data_type'] = 'alert'
        elif 'moneyflow' in result['api_name'] or 'mf' in result['api_name']:
            result['data_type'] = 'moneyflow'
        else:
            result['data_type'] = 'basic'

        return result

    def fetch_data(self, api_name: str, params: Dict) -> Optional[pd.DataFrame]:
        """调用接口获取数据"""
        if api_name not in self.api_mapping:
            print(f"⚠️ 接口 {api_name} 不可用，将尝试使用 daily")
            api_name = 'daily'
            if api_name not in self.api_mapping:
                print("❌ 没有任何可用接口")
                return None

        try:
            print(f"📊 调用接口: {api_name} - {self.api_mapping[api_name]['desc']}")
            method = self.api_mapping[api_name]['method']
            clean_params = {k: v for k, v in params.items() if v is not None and v != ""}
            print(f"  参数: {json.dumps(clean_params, ensure_ascii=False)}")

            # 特殊处理：rt_k 若没有 ts_code 则使用通配
            if api_name == 'rt_k' and 'ts_code' not in clean_params:
                clean_params['ts_code'] = '60*.SH,0*.SZ,3*.SZ'

            df = method(**clean_params)
            if df is None or df.empty:
                print(f"⚠️ 接口返回空数据")
                return None
            print(f"✅ 成功获取 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"❌ 调用失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_data(self, data_type: str, identifier: str, df: pd.DataFrame, params: Dict = None):
        """保存数据"""
        data_dir = os.path.join(self.data_dir, data_type)
        os.makedirs(data_dir, exist_ok=True)
        sub_dir = os.path.join(data_dir, identifier)
        os.makedirs(sub_dir, exist_ok=True)

        # 处理日期列
        if df is not None and not df.empty:
            for col in ['trade_date', 'trade_time', 'date', 'day', 'datetime']:
                if col in df.columns:
                    df[col] = df[col].astype(str)
                    df = df.sort_values(col)
                    break

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(sub_dir, f"{identifier}_{timestamp}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"💾 数据已保存: {csv_file}")

        # 元数据
        info_file = os.path.join(sub_dir, f"{identifier}_info.json")
        info = {
            "data_type": data_type,
            "identifier": identifier,
            "data_count": len(df),
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": params or {},
            "columns": list(df.columns)
        }
        if not df.empty:
            for col in ['trade_date', 'trade_time', 'date', 'day']:
                if col in df.columns:
                    info["date_range"] = {"first": str(df[col].min()), "last": str(df[col].max())}
                    break
            # 数值统计
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols[:5]:
                try:
                    info[f"{col}_stat"] = {
                        "max": float(df[col].max()),
                        "min": float(df[col].min()),
                        "mean": float(df[col].mean()),
                        "sum": float(df[col].sum())
                    }
                except:
                    pass
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"💾 元数据已保存: {info_file}")

        # 生成图表
        if not df.empty:
            self.plot_data(data_type, identifier, df)

    def plot_data(self, data_type: str, identifier: str, df: pd.DataFrame):
        """生成图表"""
        try:
            # 找日期列
            date_col = None
            for col in ['trade_date', 'trade_time', 'date', 'day', 'datetime']:
                if col in df.columns:
                    date_col = col
                    break
            if date_col is None:
                print("⚠️ 未找到日期列，跳过图表")
                return

            df_sorted = df.sort_values(date_col)
            df_sorted['date_str'] = df_sorted[date_col].astype(str).apply(
                lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if len(str(x)) >= 8 else str(x)
            )

            fig, ax = plt.subplots(figsize=(14, 8))

            if data_type in ['daily', 'mins', 'realtime'] and 'close' in df_sorted.columns:
                ax.plot(df_sorted['date_str'], df_sorted['close'], linewidth=2, color='#1f77b4', label='收盘价')
                ax.set_title(f'{identifier} 价格走势', fontsize=16, fontweight='bold')
                ax.set_ylabel('价格')
                ax.legend()
            elif data_type == 'sentiment':
                sentiment_cols = ['sentiment', 'score', 'value', 'hot', 'heat']
                plot_col = next((c for c in sentiment_cols if c in df_sorted.columns), None)
                if plot_col:
                    ax.plot(df_sorted['date_str'], df_sorted[plot_col], linewidth=2, color='orange', label='情绪指标')
                    ax.set_title(f'{identifier} 市场情绪', fontsize=16, fontweight='bold')
                    ax.set_ylabel('情绪值')
                    ax.legend()
            elif data_type == 'plate' and 'pct_chg' in df_sorted.columns:
                display_df = df_sorted.tail(20)
                ax.bar(display_df['date_str'], display_df['pct_chg'], alpha=0.7, color='blue')
                ax.set_title(f'{identifier} 板块涨跌幅', fontsize=16, fontweight='bold')
                ax.set_ylabel('涨跌幅(%)')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            elif data_type == 'limit':
                count_cols = ['count', 'num', 'total', 'cnt']
                plot_col = next((c for c in count_cols if c in df_sorted.columns), None)
                if plot_col:
                    ax.bar(df_sorted['date_str'], df_sorted[plot_col], alpha=0.7, color='red')
                    ax.set_title(f'{identifier} 涨停统计', fontsize=16, fontweight='bold')
                    ax.set_ylabel('数量')
            elif data_type == 'basic' and 'name' in df_sorted.columns:
                display_df = df_sorted.head(20)
                ax.barh(display_df['name'], range(len(display_df)), alpha=0.7)
                ax.set_title(f'{identifier} 数据概览', fontsize=16, fontweight='bold')
                ax.set_xlabel('索引')
            elif 'close' in df_sorted.columns:
                ax.plot(df_sorted['date_str'], df_sorted['close'], linewidth=2, color='#1f77b4')
                ax.set_title(f'{identifier} 数据走势', fontsize=16, fontweight='bold')
                ax.set_ylabel('值')
            else:
                print("⚠️ 无法绘制图表，缺少必要列")
                plt.close()
                return

            ax.grid(True, alpha=0.3)
            total_points = len(df_sorted)
            if total_points > 50:
                step = max(1, total_points // 30)
                ax.set_xticks(ax.get_xticks()[::step])
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()

            data_dir = os.path.join(self.data_dir, data_type, identifier)
            os.makedirs(data_dir, exist_ok=True)
            img_file = os.path.join(data_dir, f"{identifier}_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(img_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"📈 图表已保存: {img_file}")
        except Exception as e:
            print(f"❌ 生成图表失败: {e}")

    def process_request(self, query: str):
        """处理用户请求"""
        print("\n" + "="*60)
        print(f"📝 处理请求: {query}")
        print("="*60)

        parsed = self._parse_natural_language(query)
        api_name = parsed.get('api_name', 'daily')
        params = parsed.get('params', {})
        data_type = parsed.get('data_type', 'daily')

        print(f"\n📊 解析结果:")
        print(f"  API接口: {api_name}")
        print(f"  参数: {json.dumps(params, ensure_ascii=False, indent=2)}")
        print(f"  数据类型: {data_type}")

        df = self.fetch_data(api_name, params)
        if df is not None and not df.empty:
            identifier = f"{api_name}_{datetime.now().strftime('%Y%m%d')}"
            for key in ['ts_code', 'stock_code', 'b_code', 'plate_code']:
                if params.get(key):
                    identifier = params[key].replace('.', '_')
                    break
            self.save_data(data_type, identifier, df, params)
            print(f"\n📋 数据摘要:")
            print(f"  记录数: {len(df)}")
            print(f"  字段: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
            print("\n前5条数据:")
            print(df.head())
        else:
            print("⚠️ 未获取到数据，请检查参数或尝试其他查询。")
            print("\n💡 示例查询:")
            print("  • '获取 600871 20250101 到 20260423 的日线数据'")
            print("  • '查询 20250205 的涨停股票列表'")
            print("  • '获取 20250205 的题材板块排名'")
            print("  • '查看 20250205 的龙虎榜'")
            print("  • '获取 全部股票 列表'")

        print(f"\n✅ 处理完成！数据保存在: {os.path.abspath(self.data_dir)}")

    def run_interactive(self):
        """交互式命令行"""
        print("\n" + "="*60)
        print("🚀 股票数据智能Agent (基于zzshare)")
        print("="*60)
        print(f"\n✅ 已加载 {len(self.api_mapping)} 个可用接口")
        print("\n💡 输入自然语言查询，例如:")
        print("  • '获取 600871 20250101 到 20260423 的日线数据'")
        print("  • '查询 20250205 的涨停股票列表'")
        print("  • '获取 全部股票 列表'")
        print("  • '查看 20250205 的龙虎榜'")
        print("  • '获取 20250205 的题材板块排名'")
        print("\n输入 'exit' 退出，'help' 显示帮助")
        print("="*60)

        while self.running:
            try:
                query = input("\n📝 您: ").strip()
                if not query:
                    continue
                if query.lower() in ['exit', 'quit', 'q']:
                    print("👋 再见！")
                    break
                if query.lower() == 'help':
                    self._show_help()
                    continue
                self.process_request(query)
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
                import traceback
                traceback.print_exc()

    def _show_help(self):
        print("\n" + "="*60)
        print("📖 帮助信息")
        print("="*60)
        print("\n可用接口类别:")
        for name, info in self.api_mapping.items():
            print(f"  {name}: {info['desc']}")
        print("\n示例查询:")
        print("  行情: '获取 600871 20250101 到 20260423 的日线数据'")
        print("  涨停: '查询 20250205 的涨停股票列表'")
        print("  龙虎榜: '查看 20250205 的龙虎榜'")
        print("  板块: '查询 20250205 的题材板块排名'")
        print("  列表: '获取 全部股票 列表'")
        print("="*60)


def main():
    agent = StockDataAgent()
    agent.run_interactive()


if __name__ == "__main__":
    main()