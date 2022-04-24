import datetime
import pandas as pd
import backtrader as bt
from strategy import *

cerebro = bt.Cerebro(optreturn=False)

#Set data parameters and add to Cerebro
#获取数据
data = bt.feeds.YahooFinanceCSVData(
    dataname='C:/Users/llx/Desktop/pair_trading/代码/TSLA.csv',
    fromdate=datetime.datetime(2016, 1, 1),
    todate=datetime.datetime(2017, 3, 9))
    #settings for out-of-sample data
    #fromdate=datetime.datetime(2018, 1, 1),
    #todate=datetime.datetime(2019, 12, 25))


cerebro.adddata(data)
#Add strategy to Cerebro

#cerebro.broker.setcommission(commission=0.0002, stocklike = True)
cerebro.broker.setcash(1_0000_0000)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
cerebro.optstrategy(MAcrossover, pfast=range(1, 10), pslow=range(20, 50))

#Default position size
cerebro.addsizer(bt.sizers.SizerFix, stake=3)

if __name__ == '__main__':
    optimized_runs = cerebro.run()

    final_results_list = []
    for run in optimized_runs:
        for strategy in run:
            PnL = round(strategy.broker.get_value() - 10000, 2)
            sharpe = strategy.analyzers.sharpe_ratio.get_analysis()
            final_results_list.append([strategy.params.pfast,
                strategy.params.pslow, PnL, sharpe['sharperatio']])

    sort_by_sharpe = sorted(final_results_list, key=lambda x: x[3],
                             reverse=True)
    for line in sort_by_sharpe[:5]:
        print(line)