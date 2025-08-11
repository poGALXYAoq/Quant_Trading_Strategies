import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# --- 1. 策略核心参数 ---
# 入场相关
START_DATE_STR = '2024-05-01'
INITIAL_CAPITAL = 10_000_0  # 初始资金

# 头寸相关
FUTURES_INITIAL_LOTS = 100
PUT_OPTION_LOTS = 300

# 期权选择相关
STRIKE_PRICE_OFFSET = -200

# 网格交易相关
PRICE_GRID_INTERVAL = 10
LOTS_ADJUSTMENT_STEP = 10

# 仓位限制
FUTURES_MAX_LOTS = 300
FUTURES_MIN_LOTS = 0

# 交易成本
COMMISSION_FUTURES = 2.0  # 每手期货单边手续费
COMMISSION_OPTIONS = 1.0  # 每手期权单边手续费
SLIPPAGE = 1  # 滑点, 按价格点计算 (暂未在逻辑中实现)

# --- 2. 文件路径 ---
FUTURES_DATA_PATH = 'Protected_Futures_Reverse_Grid_Strategy/data/PTA2409/futures_data.csv'
OPTIONS_DATA_PATH = 'Protected_Futures_Reverse_Grid_Strategy/data/PTA2409/options_data.csv'
RESULTS_DIR = 'Protected_Futures_Reverse_Grid_Strategy/backtest_results'

# --- 3. 数据准备 ---
def prepare_data(futures_path, options_path):
    """加载并预处理期货和期权数据"""
    if not os.path.exists(futures_path):
        raise FileNotFoundError(f"期货数据文件未找到: {futures_path}")
    if not os.path.exists(options_path):
        raise FileNotFoundError(f"期权数据文件未找到: {options_path}")

    futures_df = pd.read_csv(futures_path)
    options_df = pd.read_csv(options_path)

    futures_df['datetime'] = pd.to_datetime(futures_df['datetime'])
    options_df['datetime'] = pd.to_datetime(options_df['datetime'])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        futures_df[col] = pd.to_numeric(futures_df[col], errors='coerce')
    for col in ['underlying_price', 'strike_price', 'close']:
        options_df[col] = pd.to_numeric(options_df[col], errors='coerce')
        
    futures_df.dropna(inplace=True)
    options_df.dropna(inplace=True)

    print(f"期货数据加载完成, {len(futures_df)} 条记录。")
    print(f"期权数据加载完成, {len(options_df)} 条记录。")
    
    return futures_df.set_index('datetime'), options_df.set_index('datetime')

# --- 4. 可视化模块 ---
def plot_results(daily_log_df, trade_log_df):
    """使用Plotly进行专业的可视化分析仪表盘"""
    if daily_log_df.empty:
        print("没有数据可供可视化。")
        return

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('期货价格与网格交易点', '账户净值曲线 (Equity Curve)', '期货持仓手数', '每日盈亏 (Daily PnL)'),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )

    # --- 子图1: 期货价格与交易信号 ---
    # K线图
    fig.add_trace(go.Candlestick(x=daily_log_df.index,
                                open=daily_log_df['futures_open'],
                                high=daily_log_df['futures_high'],
                                low=daily_log_df['futures_low'],
                                close=daily_log_df['futures_close'],
                                name='期货K线'), row=1, col=1)

    # 交易信号
    buy_signals = daily_log_df[daily_log_df['trade_action'].str.contains("buy", na=False)]
    sell_signals = daily_log_df[daily_log_df['trade_action'].str.contains("sell", na=False)]
    
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['futures_close'], mode='markers',
                             marker=dict(color='red', symbol='arrow-up', size=10, line=dict(width=2, color='DarkSlateGrey')),
                             name='买入/加仓'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['futures_close'], mode='markers',
                             marker=dict(color='limegreen', symbol='arrow-down', size=10, line=dict(width=2, color='DarkSlateGrey')),
                             name='卖出/减仓'), row=1, col=1)

    fig.update_yaxes(title_text="价格", row=1, col=1)

    # --- 子图2: 账户净值曲线 ---
    fig.add_trace(go.Scatter(x=daily_log_df.index, y=daily_log_df['total_value'], name='账户总净值',
                             line=dict(color='orange', width=2)), row=2, col=1)
    fig.update_yaxes(title_text="净值 ($)", row=2, col=1)

    # --- 子图3: 期货持仓手数 ---
    fig.add_trace(go.Scatter(x=daily_log_df.index, y=daily_log_df['futures_lots'], name='期货持仓手数',
                             fill='tozeroy', mode='lines', line=dict(color='grey', width=1)), row=3, col=1)
    fig.update_yaxes(title_text="手数", row=3, col=1)

    # --- 子图4: 每日盈亏 ---
    colors = ['limegreen' if pnl >= 0 else 'red' for pnl in daily_log_df['daily_pnl']]
    fig.add_trace(go.Bar(x=daily_log_df.index, y=daily_log_df['daily_pnl'], name='每日盈亏',
                         marker_color=colors), row=4, col=1)
    fig.update_yaxes(title_text="盈亏 ($)", row=4, col=1)
    
    # --- 整体布局设置 ---
    fig.update_layout(
        title_text='<b>受保护的期货逆势网格策略 - 回测分析报告</b>',
        height=1200,
        legend_title='图例',
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    fig.update_xaxes(title_text='日期', row=4, col=1)

    # --- 保存图表 ---
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    plot_path = os.path.join(RESULTS_DIR, 'interactive_backtest_dashboard.html')
    fig.write_html(plot_path)
    print(f"专业版交互式图表已保存至: {plot_path}")

# --- 5. 核心回测逻辑 (事件驱动) ---
def run_backtest_event_driven(futures_df, options_df):
    """执行事件驱动的策略回测（修正版）"""
    
    # --- 初始化 ---
    cash = INITIAL_CAPITAL
    futures_lots = 0
    futures_avg_price = 0.0
    options_position = {} # key: strike, value: {'lots', 'entry_price'}
    
    daily_log = []
    trade_log = []

    # 确定交易区间
    all_trading_days = futures_df.index.unique().sort_values()
    start_date = all_trading_days[all_trading_days >= pd.to_datetime(START_DATE_STR)][0]
    option_expiry_date = options_df.index.max()
    
    # 基准价格
    entry_price = futures_df.loc[start_date, 'open']
    last_total_value = INITIAL_CAPITAL

    # --- 开始每日循环 ---
    for today in all_trading_days:
        if today < start_date:
            continue
        if today > option_expiry_date:
            break

        futures_data_today = futures_df.loc[today]
        if isinstance(futures_data_today, pd.DataFrame) and not futures_data_today.empty:
            futures_data_today = futures_data_today.iloc[0] # 处理同一天有多行数据的情况
            
        futures_open = futures_data_today['open']
        futures_high = futures_data_today['high']
        futures_low = futures_data_today['low']
        futures_close = futures_data_today['close']
        
        # --- 初始化每日状态 ---
        daily_snapshot = {
            'date': today,
            'futures_open': futures_open,
            'futures_high': futures_high,
            'futures_low': futures_low,
            'futures_close': futures_close,
            'futures_lots': futures_lots,
            'trade_action': '',
            'trade_lots': 0,
            'daily_pnl': 0
        }
        
        # --- 开仓日逻辑 ---
        if today == start_date:
            # 1. 建立期货头寸
            trade_price = entry_price
            futures_lots = FUTURES_INITIAL_LOTS
            futures_avg_price = trade_price
            cost = futures_lots * trade_price
            commission = FUTURES_INITIAL_LOTS * COMMISSION_FUTURES
            cash -= (cost + commission)
            trade_log.append({'date': today, 'action': 'buy_futures_initial', 'lots': FUTURES_INITIAL_LOTS, 'price': trade_price, 'commission': commission})
            daily_snapshot.update({'trade_action': 'buy_futures_initial', 'trade_lots': FUTURES_INITIAL_LOTS})

            # 2. 建立期权保护头寸
            target_strike = entry_price + STRIKE_PRICE_OFFSET
            options_today = options_df.loc[today]
            put_options = options_today[options_today['option_type'] == 'put']
            if put_options.empty: raise ValueError(f"在 {today.date()} 找不到看跌期权")
            
            best_option = put_options.iloc[(put_options['strike_price'] - target_strike).abs().argsort()[:1]].iloc[0]
            option_price = best_option['close']
            option_strike = best_option['strike_price']

            cost = PUT_OPTION_LOTS * option_price
            commission = PUT_OPTION_LOTS * COMMISSION_OPTIONS
            cash -= (cost + commission)
            options_position[option_strike] = {'lots': PUT_OPTION_LOTS, 'entry_price': option_price}
            trade_log.append({'date': today, 'action': 'buy_put_option', 'lots': PUT_OPTION_LOTS, 'price': option_price, 'commission': commission})
            print(f"策略于 {start_date.date()} 启动。期货@{trade_price:.2f}, 期权@{option_price:.2f}(行权价{option_strike})")

        # --- 持仓调整逻辑 (非开仓日/平仓日) ---
        elif today < option_expiry_date:
            price_diff = futures_close - entry_price
            
            if price_diff > 0:
                steps = np.floor(price_diff / PRICE_GRID_INTERVAL)
                target_lots = FUTURES_INITIAL_LOTS - steps * LOTS_ADJUSTMENT_STEP
                target_lots = max(target_lots, FUTURES_MIN_LOTS)
            elif price_diff < 0:
                steps = np.floor(abs(price_diff) / PRICE_GRID_INTERVAL)
                target_lots = FUTURES_INITIAL_LOTS + steps * LOTS_ADJUSTMENT_STEP
                target_lots = min(target_lots, FUTURES_MAX_LOTS)
            else:
                target_lots = FUTURES_INITIAL_LOTS
            
            lots_to_trade = int(target_lots - futures_lots) # 确保为整数
            
            if lots_to_trade != 0:
                trade_price = futures_close
                commission = abs(lots_to_trade) * COMMISSION_FUTURES
                cash -= commission

                if lots_to_trade > 0: # 买入
                    action = 'buy_futures_grid'
                    cost = lots_to_trade * trade_price
                    cash -= cost
                    # 更新成本价
                    new_total_value = (futures_avg_price * futures_lots) + (trade_price * lots_to_trade)
                    futures_lots += lots_to_trade
                    futures_avg_price = new_total_value / futures_lots
                else: # 卖出 (lots_to_trade < 0)
                    action = 'sell_futures_grid'
                    revenue = abs(lots_to_trade) * trade_price
                    cash += revenue
                    # 平均成本价在部分卖出后通常保持不变
                    futures_lots += lots_to_trade
                
                trade_log.append({'date': today, 'action': action, 'lots': lots_to_trade, 'price': trade_price, 'commission': commission})
                daily_snapshot.update({'trade_action': action, 'trade_lots': lots_to_trade})

        # --- 平仓日逻辑 ---
        if today == option_expiry_date:
            print(f"策略于 {today.date()} 到期平仓。")
            # 1. 平期货
            if futures_lots != 0:
                revenue = futures_lots * futures_close
                cash += revenue
                commission = abs(futures_lots) * COMMISSION_FUTURES
                cash -= commission
                trade_log.append({'date': today, 'action': 'sell_futures_final', 'lots': -futures_lots, 'price': futures_close, 'commission': commission})
                daily_snapshot.update({'trade_action': 'sell_futures_final', 'trade_lots': -futures_lots})
                futures_lots = 0

            # 2. 平期权
            options_today = options_df.loc[options_df.index.isin([today])]
            for strike, pos in options_position.items():
                final_opt_price_series = options_today[options_today['strike_price'] == strike]['close']
                final_opt_price = final_opt_price_series.iloc[0] if not final_opt_price_series.empty else 0
                
                revenue = pos['lots'] * final_opt_price
                cash += revenue
                commission = pos['lots'] * COMMISSION_OPTIONS
                cash -= commission
                trade_log.append({'date': today, 'action': 'sell_option_final', 'lots': -pos['lots'], 'price': final_opt_price, 'commission': commission})
            options_position.clear()

        # --- 每日市值计算 (Mark-to-Market) ---
        # 1. 期货价值
        futures_market_value = futures_lots * futures_close

        # 2. 期权价值
        options_market_value = 0
        options_today = options_df.loc[options_df.index.isin([today])]
        if not options_today.empty:
            for strike, pos in options_position.items():
                current_opt_price_series = options_today[options_today['strike_price'] == strike]['close']
                current_opt_price = current_opt_price_series.iloc[0] if not current_opt_price_series.empty else pos['entry_price']
                options_market_value += pos['lots'] * current_opt_price
        
        # 总价值 = 现金 + 期货市值 + 期权市值
        total_value = cash + futures_market_value + options_market_value
        
        # 计算并记录每日盈亏
        daily_pnl = total_value - last_total_value
        last_total_value = total_value

        daily_snapshot['futures_lots'] = futures_lots
        daily_snapshot['total_value'] = total_value
        daily_snapshot['daily_pnl'] = daily_pnl
        daily_log.append(daily_snapshot)

    # --- 结果整合与输出 ---
    daily_log_df = pd.DataFrame(daily_log).set_index('date')
    trade_log_df = pd.DataFrame(trade_log)

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    daily_log_df.to_csv(os.path.join(RESULTS_DIR, 'daily_log.csv'))
    trade_log_df.to_csv(os.path.join(RESULTS_DIR, 'trade_log.csv'), index=False)
    
    print("\n--- 回测性能总结 ---")
    final_equity = daily_log_df['total_value'].iloc[-1]
    total_return = (final_equity / INITIAL_CAPITAL) - 1
    print(f"初始资金: {INITIAL_CAPITAL:,.2f}")
    print(f"最终权益: {final_equity:,.2f}")
    print(f"总回报率: {total_return:.2%}")
    print(f"总交易记录数: {len(trade_log_df)}")
    print("--- 结果已保存 ---")
    
    return daily_log_df, trade_log_df

if __name__ == "__main__":
    print("开始执行'受保护的期货逆势网格策略'回测 (事件驱动版)...")
    try:
        futures_data, options_data = prepare_data(FUTURES_DATA_PATH, OPTIONS_DATA_PATH)
        daily_results, trade_log = run_backtest_event_driven(futures_data, options_data)
        if daily_results is not None:
            plot_results(daily_results, trade_log)
        print("\n回测执行完毕。")
    except Exception as e:
        print(f"回测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
