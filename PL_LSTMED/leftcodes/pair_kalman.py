from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import datetime

import backtrader as bt


class KalmanMovingAverage(bt.indicators.MovingAverageBase):
    packages = ('pykalman',)
    frompackages = (('pykalman', [('KalmanFilter', 'KF')]),)
    lines = ('kma',)
    alias = ('KMA',)
    params = (
        ('initial_state_covariance', 1.0),
        ('observation_covariance', 1.0),
        ('transition_covariance', 0.05),
    )

    def __init__(self):
        self.addminperiod(self.p.period)  # when to deliver values
        self._dlast = self.data(-1)  # get previous day value

    def nextstart(self):
        self._k1 = self._dlast[0]
        self._c1 = self.p.initial_state_covariance

        self._kf = pykalman.KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            observation_covariance=self.p.observation_covariance,
            transition_covariance=self.p.transition_covariance,
            initial_state_mean=self._k1,
            initial_state_covariance=self._c1,
        )

        self.next()

    def next(self):
        k1, self._c1 = self._kf.filter_update(self._k1, self._c1, self.data[0])
        self.lines.kma[0] = self._k1 = k1

class NumPy(object):
    packages = (('numpy', 'np'),)


class KalmanFilterInd(bt.Indicator, NumPy):
    _mindatas = 2  # needs at least 2 data feeds

    packages = ('pandas',)
    lines = ('et', 'sqrt_qt')

    params = dict(
        delta=1e-4,
        vt=1e-3,
    )

    def __init__(self):
        self.wt = self.p.delta / (1 - self.p.delta) * np.eye(2)
        self.theta = np.zeros(2)
        self.R = None

        self.d1_prev = self.data1(-1)  # data1 yesterday's price

    def next(self):
        F = np.asarray([self.data0[0], 1.0]).reshape((1, 2))
        y = self.d1_prev[0]

        if self.R is not None:  # self.R starts as None, self.C set below
            self.R = self.C + self.wt
        else:
            self.R = np.zeros((2, 2))

        yhat = F.dot(self.theta)
        et = y - yhat

        # Q_t is the variance of the prediction of observations and hence
        # \sqrt{Q_t} is the standard deviation of the predictions
        Qt = F.dot(self.R).dot(F.T) + self.p.vt
        sqrt_Qt = np.sqrt(Qt)

        # The posterior value of the states \theta_t is distributed as a
        # multivariate Gaussian with mean m_t and variance-covariance C_t
        At = self.R.dot(F.T) / Qt
        self.theta = self.theta + At.flatten() * et
        self.C = self.R - At * F.dot(self.R)

        # Fill the lines
        self.lines.et[0] = et
        self.lines.sqrt_qt[0] = sqrt_Qt


class KalmanSignals(bt.Indicator):
    _mindatas = 2  # needs at least 2 data feeds

    lines = ('long', 'short',)

    def __init__(self):
        kf = KalmanFilterInd()
        et, sqrt_qt = kf.lines.et, kf.lines.sqrt_qt

        self.lines.long = et < -1.0 * sqrt_qt
        # longexit is et > -1.0 * sqrt_qt ... the opposite of long
        self.lines.short = et > sqrt_qt
        # shortexit is et < sqrt_qt ... the opposite of short


class St(bt.Strategy):
    params = dict(
        ksigs=False,  # attempt trading
        period=30,
    )

    def __init__(self):
        if self.p.ksigs:
            self.ksig = KalmanSignals()
            KalmanFilter()

        KalmanMovingAverage(period=self.p.period)
        bt.ind.SMA(period=self.p.period)
        if True:
            kf = KalmanFilterInd()
            kf.plotlines.sqrt_qt._plotskip = True

    def next(self):
        if not self.p.ksigs:
            return

        size = self.position.size
        if not size:
            if self.ksig.long:
                self.buy()
            elif self.ksig.short:
                self.sell()

        elif size > 0:
            if not self.ksig.long:
                self.close()
        elif not self.ksig.short:  # implicit size < 0
            self.close()


def runstrat(args=None):
    args = parse_args(args)

    cerebro = bt.Cerebro()

    # Data feed kwargs
    kwargs = dict()

    # Parse from/to-date
    dtfmt, tmfmt = '%Y-%m-%d', 'T%H:%M:%S'
    for a, d in ((getattr(args, x), x) for x in ['fromdate', 'todate']):
        if a:
            strpfmt = dtfmt + tmfmt * ('T' in a)
            kwargs[d] = datetime.datetime.strptime(a, strpfmt)

    # Data feed
    data0 = bt.feeds.YahooFinanceCSVData(dataname=args.data0, **kwargs)
    cerebro.adddata(data0)

    data1 = bt.feeds.YahooFinanceCSVData(dataname=args.data1, **kwargs)
    data1.plotmaster = data0
    cerebro.adddata(data1)

    # Broker
    cerebro.broker = bt.brokers.BackBroker(**eval('dict(' + args.broker + ')'))

    # Sizer
    cerebro.addsizer(bt.sizers.FixedSize, **eval('dict(' + args.sizer + ')'))

    # Strategy
    cerebro.addstrategy(St, **eval('dict(' + args.strat + ')'))

    # Execute
    cerebro.run(**eval('dict(' + args.cerebro + ')'))

    print('==================================================')
    print('Ending   Value - %.2f' % cerebro.broker.getvalue())
    print('==================================================')

    #if args.plot:  # Plot if requested to
    cerebro.plot(**eval('dict(' + args.plot + ')'))


def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            'Packages and Kalman'
        )
    )

    parser.add_argument('--data0', default='nvda-1999-2014.txt',
                        required=False, help='Data to read in')

    parser.add_argument('--data1', default='orcl-1995-2014.txt',
                        required=False, help='Data to read in')

    # Defaults for dates
    parser.add_argument('--fromdate', required=False, default='2006-01-01',
                        help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')

    parser.add_argument('--todate', required=False, default='2007-01-01',
                        help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')

    parser.add_argument('--cerebro', required=False, default='runonce=False',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--broker', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--sizer', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--strat', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--plot', required=False, default='',
                        nargs='?', const='{}',
                        metavar='kwargs', help='kwargs in key=value format')

    return parser.parse_args(pargs)


if __name__ == '__main__':
    runstrat()