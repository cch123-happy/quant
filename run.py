# strategy_agent.py
"""
AI上游右侧选股策略 Agent
功能：
1. 每日/每周对股票池进行全面分析，生成详细报告
2. 保存所有分析数据到 pool/ 文件夹
3. 交互式调仓指导：用自然语言告诉用户买卖建议
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入数据获取模块
from zzshare.client import DataApi

class StrategyAgent:
    """AI上游右侧选股策略 Agent"""
    
    def __init__(self, data_dir: str = "pool"):
        """
        初始化策略Agent
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        self._ensure_directories()
        
        # 初始化数据API
        self.api = DataApi()
        
        # 股票池
        self.stock_pool = [
    '000063', '000657', '000725', '000938', '000960', '000988', '001389', '002045',
    '002065', '002222', '002281', '002428', '002463', '002491', '002636', '002815',
    '002859', '002913', '002916', '002930', '002938', '003031', '300054', '300179',
    '300285', '300308', '300394', '300476', '300502', '300570', '300620', '300757',
    '300852', '300903', '301071', '301165', '301176', '301200', '301217', '301377',
    '301511', '600110', '600172', '600176', '600183', '600184', '600362', '600487',
    '600497', '600498', '600522', '600601', '601208', '601869', '603002', '603065',
    '603083', '603186', '603228', '603256', '603496', '603519', '603773', '603920',
    '603936', '605058', '605580', '605589', '688041', '688048', '688183', '688195',
    '688256', '688313', '688498', '688559', '688630', '688700'
]
        
        # 股票名称映射
        self.stock_names = {
    "000063": "中兴通讯", "000657": "中钨高新", "000725": "京东方A", "000938": "紫光股份",
    "000960": "锡业股份", "000988": "华工科技", "001389": "广合科技", "002045": "国光电器",
    "002065": "东华软件", "002222": "福晶科技", "002281": "光迅科技", "002428": "云南锗业",
    "002463": "沪电股份", "002491": "通鼎互联", "002636": "金安国纪", "002815": "崇达技术",
    "002859": "洁美科技", "002913": "奥士康", "002916": "深南电路", "002930": "宏川智慧",
    "002938": "鹏鼎控股", "003031": "中瓷电子", "300054": "鼎龙股份", "300179": "四方达",
    "300285": "国瓷材料", "300308": "中际旭创", "300394": "天孚通信", "300476": "胜宏科技",
    "300502": "新易盛", "300570": "太辰光", "300620": "光库科技", "300757": "罗博特科",
    "300852": "四会富仕", "300903": "科翔股份", "301071": "力量钻石", "301165": "锐捷网络",
    "301176": "逸豪新材", "301200": "大族数控", "301217": "铜冠铜箔", "301377": "鼎泰高科",
    "301511": "德福科技", "600110": "诺德股份", "600172": "黄河旋风", "600176": "中国巨石",
    "600183": "生益科技", "600184": "光电股份", "600362": "江西铜业", "600487": "亨通光电",
    "600497": "驰宏锌锗", "600498": "烽火通信", "600522": "中天科技", "600601": "方正科技",
    "601208": "东材科技", "601869": "长飞光纤", "603002": "宏昌电子", "603065": "宿迁联盛",
    "603083": "剑桥科技", "603186": "华正新材", "603228": "景旺电子", "603256": "宏和科技",
    "603496": "恒为科技", "603519": "立霸股份", "603773": "沃格光电", "603920": "世运电路",
    "603936": "博敏电子", "605058": "澳弘电子", "605580": "恒盛能源", "605589": "圣泉集团",
    "688041": "海光信息", "688048": "长光华芯", "688183": "生益电子", "688195": "腾景科技",
    "688256": "寒武纪", "688313": "仕佳光子", "688498": "源杰科技", "688559": "海目星",
    "688630": "芯碁微装", "688700": "东威科技"
}
        
        # 当前持仓（模拟）
        self.current_positions = {}  # {stock_code: shares}
        self.buy_price = {}  # {stock_code: buy_price}
        self.cash = 1000000  # 初始资金100万
        
        # 运行状态
        self.running = True
        
        print("🚀 AI上游右侧选股策略 Agent 初始化完成")
        print(f"📊 股票池：{len(self.stock_pool)} 只股票")
        print(f"📁 数据目录：{os.path.abspath(self.data_dir)}")
        
    def _ensure_directories(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # 创建子目录
        sub_dirs = ['reports', 'charts', 'data', 'backtest']
        for sub in sub_dirs:
            sub_path = os.path.join(self.data_dir, sub)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
                
    def _get_stock_code_with_suffix(self, code: str) -> str:
        """获取带交易所后缀的股票代码"""
        if code.startswith('6'):
            return f"{code}.SH"
        elif code.startswith('0') or code.startswith('3'):
            return f"{code}.SZ"
        else:
            return f"{code}.SH"
            
    def fetch_stock_data(self, stock_code: str, days: int = 120) -> Optional[pd.DataFrame]:
        """获取单只股票历史数据"""
        try:
            ts_code = self._get_stock_code_with_suffix(stock_code)
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
            
            max_retries = 5
            retry_delay = 60
            
            for attempt in range(max_retries):
                try:
                    df = self.api.daily(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date,
                        fields='all'
                    )
                    
                    if df is not None and not df.empty:
                        if 'trade_date' in df.columns:
                            df['trade_date'] = pd.to_numeric(df['trade_date'])
                            today_int = int(datetime.now().strftime('%Y%m%d'))
                            df = df[df['trade_date'] <= today_int]
                            df = df.sort_values('trade_date').reset_index(drop=True)
                        
                        # 修复：统一字段名为 'vol'
                        if 'vol' not in df.columns and 'volume' in df.columns:
                            df['vol'] = df['volume']
                        
                        return df
                    else:
                        return None
                        
                except Exception as e:
                    error_msg = str(e)
                    if '429' in error_msg or 'Rate limit' in error_msg:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"  ⚠️ 触发限流 (尝试 {attempt+1}/{max_retries})，等待 {wait_time:.0f}秒...")
                        time.sleep(wait_time)
                    else:
                        print(f"  ❌ 获取 {stock_code} 失败: {e}")
                        return None
                        
            print(f"  ❌ {stock_code} 重试 {max_retries} 次后仍失败")
            return None
            
        except Exception as e:
            print(f"❌ 获取 {stock_code} 数据失败: {e}")
            return None
            
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """计算技术指标"""
        if df is None or df.empty:
            return {}
            
        # 检查必需字段
        if 'close' not in df.columns:
            return {}
        
        # 修复：统一成交量字段名
        if 'vol' not in df.columns:
            if 'volume' in df.columns:
                df['vol'] = df['volume']
            else:
                # 如果没有成交量数据，使用默认值
                df['vol'] = 1
            
        close = df['close'].values
        volume = df['vol'].values
        
        if len(close) < 30:
            return {}
            
        # 均线
        ma10 = np.mean(close[-10:])
        ma20 = np.mean(close[-20:])
        ma60 = np.mean(close[-60:]) if len(close) >= 60 else ma20
        
        # 成交量均线
        vol_ma5 = np.mean(volume[-5:])
        vol_ma20 = np.mean(volume[-20:]) if len(volume) >= 20 else vol_ma5
        
        # 当前价格
        current_price = close[-1]
        
        # 涨跌幅
        if len(close) >= 2:
            day_change = (close[-1] / close[-2] - 1) * 100
        else:
            day_change = 0
            
        if len(close) >= 20:
            month_change = (close[-1] / close[-20] - 1) * 100
        else:
            month_change = 0
            
        # 最高最低
        high_60 = np.max(close[-60:]) if len(close) >= 60 else np.max(close)
        low_60 = np.min(close[-60:]) if len(close) >= 60 else np.min(close)
        
        return {
            'current_price': current_price,
            'ma10': ma10,
            'ma20': ma20,
            'ma60': ma60,
            'vol_ma5': vol_ma5,
            'vol_ma20': vol_ma20,
            'day_change': day_change,
            'month_change': month_change,
            'high_60': high_60,
            'low_60': low_60,
            'data_count': len(close)
        }
        
    def evaluate_stock(self, stock_code: str, indicators: Dict) -> Dict:
        """评估股票是否符合策略"""
        if not indicators:
            return {
                'stock_code': stock_code,
                'stock_name': self.stock_names.get(stock_code, stock_code),
                'score': 0,
                'selected': False,
                'reasons': ['数据不足']
            }
            
        close = indicators['current_price']
        ma10 = indicators['ma10']
        ma20 = indicators['ma20']
        ma60 = indicators['ma60']
        vol_ma5 = indicators['vol_ma5']
        vol_ma20 = indicators['vol_ma20']
        
        # ===== 选股条件 =====
        reasons = []
        conditions = []
        score = 0
        
        # 条件1：价格 > MA20 * 0.92
        if close > ma20 * 0.92:
            conditions.append(True)
            reasons.append(f'价格在MA20附近 ({close:.2f} > {ma20*0.92:.2f})')
            score += 20
        else:
            conditions.append(False)
            reasons.append(f'价格低于MA20的92%')
        
        # 条件2：成交量放大（修复除零错误）
        if vol_ma20 > 0 and vol_ma5 > vol_ma20 * 1.05:
            conditions.append(True)
            reasons.append(f'成交量放大 {vol_ma5/vol_ma20:.2f}倍')
            score += 20
        else:
            conditions.append(False)
            if vol_ma20 > 0:
                reasons.append(f'成交量未明显放大 ({vol_ma5/vol_ma20:.2f}倍)')
            else:
                reasons.append('成交量数据不足')
        
        # 条件3：趋势向上
        trend_conditions = []
        if ma10 > ma20:
            trend_conditions.append('MA10 > MA20')
            score += 15
        if ma20 > ma60:
            trend_conditions.append('MA20 > MA60')
            score += 15
        if close > ma60 * 1.01:
            trend_conditions.append('价格 > MA60')
            score += 10
            
        if trend_conditions:
            conditions.append(True)
            reasons.append(f'趋势向上: {", ".join(trend_conditions)}')
        else:
            conditions.append(False)
            reasons.append('趋势偏弱')
        
        # ===== 综合评分 =====
        trend_score = 0
        if ma10 > ma20:
            trend_score += 30
        if ma20 > ma60:
            trend_score += 30
        if close > ma60:
            trend_score += 20 * (close / ma60 - 1)
        
        # 成交量得分（修复除零）
        if vol_ma20 > 0:
            vol_ratio = vol_ma5 / vol_ma20
            if vol_ratio > 1.5:
                vol_score = 30
            elif vol_ratio > 1.2:
                vol_score = 20
            elif vol_ratio > 1.0:
                vol_score = 10
            else:
                vol_score = 0
        else:
            vol_score = 0
        
        # 价格位置得分
        if ma20 > 0:
            price_position = close / ma20
            if price_position > 1.1:
                position_score = 20
            elif price_position > 1.0:
                position_score = 15
            elif price_position > 0.95:
                position_score = 10
            else:
                position_score = 0
        else:
            position_score = 0
        
        total_score = trend_score + vol_score + position_score
        
        # ===== 入选条件 =====
        passed_conditions = sum(conditions)
        selected = passed_conditions >= 2 and total_score > 30
        
        return {
            'stock_code': stock_code,
            'stock_name': self.stock_names.get(stock_code, stock_code),
            'current_price': close,
            'ma10': ma10,
            'ma20': ma20,
            'ma60': ma60,
            'vol_ma5': vol_ma5,
            'vol_ma20': vol_ma20,
            'day_change': indicators.get('day_change', 0),
            'month_change': indicators.get('month_change', 0),
            'score': total_score,
            'selected': selected,
            'conditions': conditions,
            'conditions_passed': passed_conditions,
            'reasons': reasons,
            'data_count': indicators.get('data_count', 0)
        }
        
    def analyze_pool(self) -> pd.DataFrame:
        """
        分析整个股票池
        
        Returns:
            分析结果DataFrame
        """
        print("\n" + "="*70)
        print(f"📊 开始分析股票池 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        results = []
        
        for i, code in enumerate(self.stock_pool):
            print(f"\r🔄 分析进度: {i+1}/{len(self.stock_pool)} - {code}", end="")
            
            # 获取数据
            df = self.fetch_stock_data(code, days=120)
            if df is None or df.empty:
                results.append({
                    'stock_code': code,
                    'stock_name': self.stock_names.get(code, code),
                    'selected': False,
                    'score': 0,
                    'reasons': ['数据获取失败']
                })
                continue
                
            # 计算指标
            indicators = self.calculate_indicators(df)
            if not indicators:
                results.append({
                    'stock_code': code,
                    'stock_name': self.stock_names.get(code, code),
                    'selected': False,
                    'score': 0,
                    'reasons': ['数据不足']
                })
                continue
                
            # 评估
            evaluation = self.evaluate_stock(code, indicators)
            results.append(evaluation)
            
        print("\n✅ 分析完成！")
        
        # 转换为DataFrame
        df_results = pd.DataFrame(results)
        
        # 按得分排序
        df_results = df_results.sort_values('score', ascending=False).reset_index(drop=True)
        
        return df_results
        
    def generate_report(self, df_results: pd.DataFrame) -> Dict:
        """
        生成完整报告
        
        Args:
            df_results: 分析结果DataFrame
            
        Returns:
            报告字典
        """
        # 筛选入选股票
        selected_stocks = df_results[df_results['selected'] == True]
        
        # 构建报告
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_stocks': len(df_results),
            'selected_count': len(selected_stocks),
            'selected_stocks': selected_stocks.to_dict('records'),
            'all_stocks': df_results.to_dict('records'),
            'top_5': df_results.head(5).to_dict('records'),
            'summary': {
                'avg_score': df_results['score'].mean(),
                'max_score': df_results['score'].max(),
                'min_score': df_results['score'].min(),
                'selected_names': selected_stocks['stock_name'].tolist() if not selected_stocks.empty else []
            }
        }
        
        # 保存报告到文件
        report_file = os.path.join(self.data_dir, 'reports', 
                                   f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        # 保存CSV
        csv_file = os.path.join(self.data_dir, 'data', 
                                f"pool_analysis_{datetime.now().strftime('%Y%m%d')}.csv")
        df_results.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # 保存入选股票列表
        if not selected_stocks.empty:
            selected_file = os.path.join(self.data_dir, 'data', 
                                         f"selected_{datetime.now().strftime('%Y%m%d')}.csv")
            selected_stocks.to_csv(selected_file, index=False, encoding='utf-8-sig')
            
        # 生成文本报告
        report['text_report'] = self._generate_text_report(df_results, selected_stocks)
        
        print(f"\n📁 报告已保存: {report_file}")
        print(f"📁 CSV已保存: {csv_file}")
        
        return report
        
    def _generate_text_report(self, df_results: pd.DataFrame, selected_stocks: pd.DataFrame) -> str:
        """生成文本格式报告"""
        lines = []
        lines.append("="*70)
        lines.append(f"📊 AI上游右侧选股策略 - 分析报告")
        lines.append(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("="*70)
        
        # 入选股票
        lines.append(f"\n🎯 入选股票 ({len(selected_stocks)}只):")
        if not selected_stocks.empty:
            for _, row in selected_stocks.iterrows():
                lines.append(f"  ✅ {row['stock_code']} {row['stock_name']} - 得分: {row['score']:.2f}")
                lines.append(f"     价格: {row['current_price']:.2f} | MA20: {row['ma20']:.2f} | MA60: {row['ma60']:.2f}")
        else:
            lines.append("  ⚠️ 当前无股票符合选股条件")
            
        # Top 5
        lines.append(f"\n📈 综合得分 Top 5:")
        for i, row in df_results.head(5).iterrows():
            status = "✅" if row['selected'] else "❌"
            lines.append(f"  {status} {row['stock_code']} {row['stock_name']} - 得分: {row['score']:.2f}")
            lines.append(f"     当前价: {row['current_price']:.2f} | 涨跌幅: {row.get('day_change', 0):.2f}%")
            
        # 统计信息
        lines.append("\n" + "-"*70)
        lines.append("📊 统计信息:")
        lines.append(f"  总股票数: {len(df_results)}")
        lines.append(f"  入选股票数: {len(selected_stocks)}")
        lines.append(f"  入选率: {len(selected_stocks)/len(df_results)*100:.1f}%")
        
        # 评分分布
        score_bins = [0, 10, 20, 30, 40, 50, 100]
        score_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50+']
        df_results['score_group'] = pd.cut(df_results['score'], bins=score_bins, labels=score_labels)
        score_dist = df_results['score_group'].value_counts().sort_index()
        lines.append(f"\n  评分分布:")
        for group, count in score_dist.items():
            lines.append(f"    {group}: {count}只")
            
        lines.append("="*70)
        
        return "\n".join(lines)
        
    def generate_trading_advice(self, selected_stocks: List[str], 
                            current_positions: Dict = None) -> Dict:
        """生成交易建议 - 修复股数显示问题"""
        if current_positions is None:
            current_positions = self.current_positions
        
        # 如果持仓为空，自动创建演示持仓
        if not current_positions:
            demo_positions = {}
            for stock in selected_stocks[:3]:
                demo_positions[stock] = 1000  # 每只1000股
            current_positions = demo_positions
            self.current_positions = demo_positions
            print(f"  ℹ️ 已初始化演示持仓: {len(demo_positions)} 只")
        
        target_stocks = selected_stocks[:5]
        
        advice = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'target_stocks': target_stocks,
            'current_positions': current_positions,
            'buy': [],
            'sell': [],
            'hold': [],
            'actions': []
        }
        
        # 需要卖出的股票
        for stock in current_positions.keys():
            if stock not in target_stocks:
                advice['sell'].append({
                    'stock_code': stock,
                    'stock_name': self.stock_names.get(stock, stock),
                    'shares': current_positions[stock],
                    'action': '卖出全部'
                })
                advice['actions'].append(f"📤 卖出 {stock} {self.stock_names.get(stock, stock)} ({current_positions[stock]}股)")
        
        # 需要买入的股票
        for stock in target_stocks:
            if stock not in current_positions:
                advice['buy'].append({
                    'stock_code': stock,
                    'stock_name': self.stock_names.get(stock, stock),
                    'action': '买入'
                })
                advice['actions'].append(f"📥 买入 {stock} {self.stock_names.get(stock, stock)}")
            else:
                advice['hold'].append({
                    'stock_code': stock,
                    'stock_name': self.stock_names.get(stock, stock),
                    'shares': current_positions[stock],
                    'action': '持有'
                })
                advice['actions'].append(f"✅ 持有 {stock} {self.stock_names.get(stock, stock)} ({current_positions[stock]}股)")
        
        advice['natural_language'] = self._generate_natural_language(advice)
        
        return advice
        
    def _generate_natural_language(self, advice: Dict) -> str:
        """生成自然语言交易建议"""
        lines = []
        lines.append("📋 调仓建议")
        lines.append("="*60)
        
        if advice['buy'] and advice['sell']:
            lines.append("🔄 需要调仓操作：")
            lines.append(f"   📤 卖出 {len(advice['sell'])} 只")
            for s in advice['sell']:
                lines.append(f"      - {s['stock_code']} {s['stock_name']}")
            lines.append(f"   📥 买入 {len(advice['buy'])} 只")
            for s in advice['buy']:
                lines.append(f"      - {s['stock_code']} {s['stock_name']}")
        elif advice['buy'] and not advice['sell']:
            lines.append("📥 新增持仓：")
            for s in advice['buy']:
                lines.append(f"   - 买入 {s['stock_code']} {s['stock_name']}")
        elif advice['sell'] and not advice['buy']:
            lines.append("📤 减仓操作：")
            for s in advice['sell']:
                lines.append(f"   - 卖出 {s['stock_code']} {s['stock_name']}")
        else:
            lines.append("✅ 当前持仓符合策略，无需调仓")
            
        # 目标持仓
        lines.append("\n🎯 目标持仓 (最多5只):")
        for stock in advice['target_stocks']:
            name = self.stock_names.get(stock, stock)
            if stock in advice['current_positions']:
                shares = advice['current_positions'][stock]
                lines.append(f"   - {stock} {name} (持有{shares}股)")
            else:
                lines.append(f"   - {stock} {name} (待买入)")
                
        return "\n".join(lines)
        
    def get_trading_decisions(self, user_query: str = "") -> Dict:
        """
        获取交易决策 - 增加数据诊断
        """
        print("\n🔍 开始数据诊断...")
        
        # 先测试一只股票的数据获取
        test_code = '600330'
        print(f"\n测试获取 {test_code} 数据...")
        test_df = self.fetch_stock_data(test_code, days=30)
        if test_df is not None and not test_df.empty:
            print(f"✅ 数据获取成功，最新日期: {test_df['trade_date'].iloc[-1]}")
            print(f"   最新收盘价: {test_df['close'].iloc[-1]}")
            print(f"   数据列: {test_df.columns.tolist()}")
        else:
            print("❌ 数据获取失败，请检查 zzshare 配置")
            return {'report': None, 'advice': None, 'df_results': None}
        
        print("\n" + "="*70)
        print("📊 开始分析股票池...")
        print("="*70)
        
        # 1. 分析股票池
        df_results = self.analyze_pool()
        
        if df_results is None or df_results.empty:
            print("❌ 分析结果为空")
            return {'report': None, 'advice': None, 'df_results': None}
        
        # 2. 生成报告
        report = self.generate_report(df_results)
        
        # 3. 获取选中股票（放宽：得分>50且至少满足2个条件）
        df_results['selected'] = (df_results['score'] > 50) & (df_results.get('conditions_passed', 0) >= 2)
        
        # 重新筛选
        selected_stocks = df_results[df_results['selected'] == True]['stock_code'].tolist()
        
        # 如果还是没有选中，选择得分最高的前3只作为候选
        if not selected_stocks:
            print("\n⚠️ 无股票满足所有条件，使用得分最高的前3只作为候选")
            selected_stocks = df_results.head(3)['stock_code'].tolist()
            # 标记为候选
            df_results.loc[df_results['stock_code'].isin(selected_stocks), 'selected'] = True
        
        # 4. 生成交易建议
        advice = self.generate_trading_advice(selected_stocks)
        
        return {
            'report': report,
            'advice': advice,
            'df_results': df_results
        }
        
    def print_trading_report(self, decisions: Dict):
        """打印交易报告"""
        report = decisions['report']
        advice = decisions['advice']
        
        print("\n" + "="*70)
        print("📊 策略分析及交易建议")
        print("="*70)
        
        # 打印文本报告
        print(report['text_report'])
        
        # 打印交易建议
        print("\n" + advice['natural_language'])
        
        print("\n" + "="*70)
        
    def run_analysis_mode(self):
        """运行分析模式 - 一次性分析并显示结果"""
        print("\n🚀 开始策略分析...")
        decisions = self.get_trading_decisions()
        self.print_trading_report(decisions)
        
        # 保存当前选股结果
        selected = decisions['advice']['target_stocks']
        print(f"\n💾 分析结果已保存到 {self.data_dir}/")
        print(f"🎯 当前建议持仓: {selected}")
        
        return decisions
        
    def run_interactive(self):
        """运行交互式命令行界面"""
        print("\n" + "="*70)
        print("🚀 AI上游右侧选股策略 Agent - 交互式模式")
        print("="*70)
        print("\n💡 使用说明:")
        print("  1. 输入 'analyze' - 运行策略分析并查看报告")
        print("  2. 输入 'advice'  - 查看当前交易建议")
        print("  3. 输入 'positions' - 查看当前持仓")
        print("  4. 输入 'update' - 更新持仓（手动输入买卖）")
        print("  5. 输入 'help' - 显示帮助")
        print("  6. 输入 'exit' - 退出程序")
        print("\n" + "="*70)
        
        # 存储最近的决策结果
        last_decisions = None
        
        while self.running:
            try:
                query = input("\n📝 您: ").strip().lower()
                
                if not query:
                    continue
                    
                if query in ['exit', 'quit', 'q']:
                    print("👋 再见！")
                    break
                    
                if query == 'help':
                    self._show_help()
                    continue
                    
                if query == 'analyze':
                    last_decisions = self.run_analysis_mode()
                    continue
                    
                if query == 'advice':
                    if last_decisions is not None:
                        self.print_trading_report(last_decisions)
                    else:
                        print("⚠️ 请先运行 'analyze' 进行分析")
                    continue
                    
                if query == 'positions':
                    self._show_positions()
                    continue
                    
                if query == 'update':
                    self._update_positions_interactive()
                    continue
                    
                # 自然语言查询 - 解析买卖建议
                if any(word in query for word in ['买', '卖', '持有', '调仓']):
                    if last_decisions is not None:
                        advice = last_decisions['advice']
                        print("\n" + advice['natural_language'])
                    else:
                        print("⚠️ 请先运行 'analyze' 进行分析")
                    continue
                    
                print(f"❌ 未知命令: {query}，输入 'help' 查看帮助")
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 处理请求时发生错误: {e}")
                
    def _show_positions(self):
        """显示当前持仓"""
        print("\n" + "="*60)
        print("📊 当前持仓")
        print("="*60)
        
        if not self.current_positions:
            print("  ⚠️ 当前无持仓")
            return
            
        total_value = 0
        for stock, shares in self.current_positions.items():
            name = self.stock_names.get(stock, stock)
            # 获取最新价格
            df = self.fetch_stock_data(stock, days=10)
            if df is not None and not df.empty:
                price = df['close'].iloc[-1]
                value = price * shares
                total_value += value
                print(f"  {stock} {name}: {shares}股, 价格: {price:.2f}, 市值: {value:.2f}")
            else:
                print(f"  {stock} {name}: {shares}股 (无法获取价格)")
                
        print(f"\n  总持仓市值: {total_value:.2f}")
        print("="*60)
        
    def _update_positions_interactive(self):
        """交互式更新持仓"""
        print("\n" + "="*60)
        print("📝 更新持仓")
        print("="*60)
        print("输入格式: 股票代码 数量 (如: 600330 1000)")
        print("输入 'done' 完成更新")
        print("输入 'clear' 清空所有持仓")
        print("="*60)
        
        while True:
            user_input = input("> ").strip()
            
            if user_input.lower() == 'done':
                break
                
            if user_input.lower() == 'clear':
                self.current_positions = {}
                print("✅ 已清空所有持仓")
                continue
                
            parts = user_input.split()
            if len(parts) == 2:
                code = parts[0]
                try:
                    shares = int(parts[1])
                    if code in self.stock_pool:
                        if shares > 0:
                            self.current_positions[code] = shares
                            print(f"✅ 已更新 {code} {self.stock_names.get(code, code)}: {shares}股")
                        else:
                            if code in self.current_positions:
                                del self.current_positions[code]
                                print(f"✅ 已移除 {code}")
                    else:
                        print(f"⚠️ {code} 不在股票池中")
                except ValueError:
                    print("⚠️ 请输入有效的数量")
            else:
                print("⚠️ 格式错误，请使用: 股票代码 数量")
                
        print("\n✅ 持仓已更新")
        self._show_positions()
        
    def _show_help(self):
        """显示帮助"""
        print("\n" + "="*60)
        print("📖 帮助信息")
        print("="*60)
        print("\n可用命令:")
        print("  analyze    - 运行策略分析，生成完整报告")
        print("  advice     - 查看当前交易建议（需先运行analyze）")
        print("  positions  - 查看当前持仓")
        print("  update     - 手动更新持仓")
        print("  help       - 显示此帮助")
        print("  exit       - 退出程序")
        print("\n自然语言示例:")
        print("  '买什么股票'  - 查看买入建议")
        print("  '卖哪些'      - 查看卖出建议")
        print("  '调仓'        - 查看完整调仓建议")
        print("="*60 + "\n")
        
    def set_positions(self, positions: Dict):
        """设置持仓（外部调用）"""
        self.current_positions = positions
        print(f"✅ 已更新持仓: {len(positions)} 只股票")


def main():
    """主函数"""
    # 创建Agent
    agent = StrategyAgent()
    
    # 检查是否有命令行参数
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'analyze':
            agent.run_analysis_mode()
            return
        elif sys.argv[1] == 'interactive':
            agent.run_interactive()
            return
            
    # 默认运行交互式模式
    agent.run_interactive()


if __name__ == "__main__":
    main()