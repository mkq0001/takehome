from numpy import ndarray
from typing import Tuple, Dict
from pandas import Index, DataFrame
from argparse import ArgumentParser


def make_parser() -> ArgumentParser:
    """
        parse command line information
    """
    ans = ArgumentParser(description='main executable')
    ans.add_argument('--signal-scale', dest='signal_scale', default=2e-4, type=float,
                     help='overall control of returns predictability by the features, this is roughly the r2 of '
                          'each individual feature')
    ans.add_argument('--fake-sig-upscale', dest='fake_sig_upscale', default=3., type=float,
                     help='for the signal which breaks out-of-sample, this is the factor to upscale its in-sample '
                          'performance to make it more appealing than others.')
    ans.add_argument('--fake-sig-fraction', dest='fake_sig_fraction', default=.3, type=float,
                     help='for the signal which breaks out-of-sample, this is the fraction of the out-sample length')
    ans.add_argument('--factor-trend-scale', dest='factor_trend_scale', default=.1, type=float,
                     help='relative scale of how trending factors are compared with the variance of factor returns')
    ans.add_argument('--seed', dest='seed', default=34, type=int,
                     help='relative scale of how trending factors are compared with the variance of factor returns')
    ans.add_argument('--location', dest='location', type=str, help='output location for the csv.gz files')
    ans.add_argument('--debug', action='store_true', help='reveal the model instead of saving the files')
    return ans


class Problem(object):
    """

    """
    @property
    def num_date(self) -> int: return int(self.date_start + self.num_asset * 2 + 1)

    @property
    def date_start(self) -> int:
        """

        """
        ans = [self.factor_window, self.spec_rev_window, self.slow_sig_window, self.fast_sig_window]
        return max(ans)

    @property
    def date_end(self) -> int: return self.num_date - 1

    @property
    def num_asset(self) -> int: return 200

    @property
    def percent_missing(self) -> float: return .01

    @property
    def num_signal(self) -> int: return 10

    @property
    def num_factor(self) -> int: return 3

    @property
    def factor_window(self) -> int: return 250

    @property
    def spec_volatility(self) -> float: return .02

    @property
    def factor_volatility(self) -> float: return .02

    @property
    def spec_rev_window(self) -> int: return 3

    @property
    def factor_trend_window(self) -> int: return 5

    @property
    def slow_sig_window(self) -> int: return 20

    @property
    def fast_sig_window(self) -> int: return 2

    @property
    def cond_sig_window(self) -> int: return 3

    @property
    def fake_sig_window(self) -> int: return 20

    @property
    def columns(self) -> Index:
        """

        """
        ans = [f'id_{idx:03d}' for idx in range(self.num_asset)]
        return Index(data=ans, name='id')

    @property
    def index(self) -> Index:
        """

        """
        ans = [f'd_{idx - self.date_start:04d}' for idx in range(self.num_date)]
        return Index(data=ans, name='date')

    @staticmethod
    def smooth(arr: ndarray, window: int, shift: int) -> ndarray:
        """

        """
        from scipy.ndimage import convolve1d
        from numpy import ones, zeros, concatenate

        # moving average, gap one observation
        weight = ones(shape=window) / window
        weight = concatenate([zeros(shape=window), weight], axis=0)
        ans = convolve1d(input=arr, weights=weight, axis=0, output=None, mode='constant', cval=0., origin=-shift)
        return ans

    @staticmethod
    def add_component(arr: ndarray, other: ndarray, scale: float) -> ndarray:
        """

        """
        from numpy import sqrt, sign

        # scale other
        ratio = arr.var(axis=None) * scale / other.var(axis=None)
        ratio = sqrt(abs(ratio)) * sign(scale)
        ans = arr + other * ratio
        return ans

    def __init__(self, signal_scale: float, fake_sig_upscale: float, fake_sig_fraction: float, factor_trend_scale: float) -> None:
        """

        """
        self.signal_scale = signal_scale
        self.fake_sig_upscale = fake_sig_upscale
        self.fake_sig_fraction = fake_sig_fraction
        self.factor_trend_scale = factor_trend_scale

    def normal(self, scale: float, dim: int) -> ndarray:
        """

        """
        from numpy.random import normal

        ans = normal(loc=0., scale=scale, size=[self.num_date, dim])
        return ans

    def get_exposure(self) -> Dict[int, ndarray]:
        """

        """
        # create factor exposure
        ans = dict()
        for fdx in range(self.num_factor):
            value = self.normal(scale=1., dim=self.num_asset)
            value = self.smooth(arr=value, window=self.factor_window, shift=0)
            ans[fdx] = value
        return ans

    def get_factor_return(self) -> ndarray:
        """

        """
        value = self.normal(scale=self.factor_volatility, dim=self.num_factor)
        other = self.smooth(arr=value, window=self.factor_trend_window, shift=1)
        ans = self.add_component(arr=value, other=other, scale=self.factor_trend_scale)
        return ans

    def compute(self, debug: bool) -> Tuple[DataFrame, DataFrame]:
        """

        """
        from pandas import concat
        from numpy.random import choice

        # factor exposure and factor returns
        factor, factor_return = self.get_exposure(), self.get_factor_return()
        # specific random returns
        data = self.normal(scale=self.spec_volatility, dim=self.num_asset)

        # add specific returns reversal
        data, _ = self.add_reversal(data=data, scale=self.signal_scale)
        # add slow signal
        data, slow = self.add_signal(data=data, window=self.slow_sig_window, scale=self.signal_scale)
        # add slow signal
        data, fast = self.add_signal(data=data, window=self.fast_sig_window, scale=self.signal_scale)
        # add condition signal
        use_fdx = choice(range(self.num_factor))
        data, cond = self.add_condition(data=data, factor=factor[use_fdx], scale=self.signal_scale)
        # add a signal which over-fits
        window, frac, scale = self.fake_sig_window, self.fake_sig_fraction, self.fake_sig_upscale * self.signal_scale
        data, fake, fake_ = self.add_broken(data=data, window=window, scale=scale, fraction=frac)

        # add factor trend
        factor_sum, factor_return = 0., self.get_factor_return()
        for index in range(self.num_factor):
            value = factor[index] * factor_return[:, index].reshape([-1, 1])
            factor_sum = factor_sum + value
        data = factor_sum + data

        # package the output
        f_data = list()
        for index, value in factor.items():
            value = DataFrame(data=value / value.std(axis=None), columns=self.columns, index=self.index)
            value = value.stack().rename(f'f_{index}')
            f_data.append(value)
        f_data = concat(f_data, axis=1)
        data = DataFrame(data=data, columns=self.columns, index=self.index)
        slow = DataFrame(data=slow / slow.std(axis=None), columns=self.columns, index=self.index)
        fast = DataFrame(data=fast / fast.std(axis=None), columns=self.columns, index=self.index)
        cond = DataFrame(data=cond / cond.std(axis=None), columns=self.columns, index=self.index)
        fake = DataFrame(data=fake / fake.std(axis=None), columns=self.columns, index=self.index)
        x_data = [
            slow.stack().rename('slow'),
            fast.stack().rename('fast'),
            cond.stack().rename('cond'),
            fake.stack().rename('fake'),
        ]
        for index in range(self.num_signal - len(f_data.columns) - len(x_data)):
            window = choice(range(1, self.spec_rev_window))
            value = self.normal(scale=1., dim=self.num_asset)
            value = self.smooth(arr=value, window=window, shift=0)
            value = DataFrame(data=value / value.std(), columns=self.columns, index=self.index)
            x_data.append(value.stack().rename(f'noise{index}_window{window}'))
        x_data = concat(x_data + [f_data], axis=1)

        if debug:
            from numpy.linalg import pinv as invert
            from statsmodels.regression.linear_model import OLS

            # test a model
            fake_ = DataFrame(data=fake_, columns=self.columns, index=self.index)
            revs = data.rolling(window=self.spec_rev_window, axis=0).mean().shift(1).fillna(value=0.)
            cond_ = cond.stack() * f_data[f'f_{use_fdx}']
            # solve factor returns
            factor_return2 = factor_return * 0.
            for ddx in range(self.num_date):
                key = f'd_{ddx - self.date_start:04d}'
                B = f_data.loc[key]
                y = data.loc[key]
                kernel = invert(B.T @ B)
                f = kernel @ B.T @ y
                factor_return2[ddx, :] = f
            columns = [f'f_{fdx}' for fdx in range(self.num_factor)]
            fret = DataFrame(data=factor_return2, index=self.index, columns=columns)
            fret = fret.rolling(window=self.factor_trend_window, axis=0).mean().shift(1).fillna(value=0.)
            ftr2 = list()
            for fdx, fcol in enumerate(f_data.columns):
                fval = f_data[fcol].unstack()
                fsig = fval.mul(fret.values[:, fdx], axis=0).stack().rename(f'ftr_{fdx}')
                ftr2.append(fsig)
            ftr2 = concat(ftr2, axis=1,)
            x_data = [
                revs.iloc[self.date_start:self.date_end].stack().rename(f'lag return smooth {self.spec_rev_window}d'),
                slow.iloc[self.date_start:self.date_end].stack().rename('slow signal'),
                fast.iloc[self.date_start:self.date_end].stack().rename('fast signal'),
                cond.iloc[self.date_start:self.date_end].stack().rename('plain conditioner'),
                cond_.unstack().iloc[self.date_start:self.date_end].stack().rename('conditioner applied on factor'),
                fake.iloc[self.date_start:self.date_end].stack().rename('fake signal after masking'),
                fake_.iloc[self.date_start:self.date_end].stack().rename('fake signal before masking'),
                ftr2.unstack().iloc[self.date_start:self.date_end].stack(),
            ]
            x_data = concat(x_data + [f_data.unstack().iloc[self.date_start:self.date_end].stack()], axis=1)
            y_data = data.iloc[self.date_start:self.date_end].stack().rename('label')
            print(OLS(endog=y_data, exog=x_data).fit().summary())

        return x_data, data

    def add_reversal(self, data: ndarray, scale: float) -> Tuple[ndarray, ndarray]:
        """

        """
        # idiosyncratic returns with a bit reversal
        value = self.smooth(arr=data, window=self.spec_rev_window, shift=1)
        data = self.add_component(arr=data, other=value, scale=-1 * scale)
        return data, value

    def add_signal(self, data: ndarray, window: int, scale: float) -> Tuple[ndarray, ndarray]:
        """

        """
        # add a signal with scale 1.
        value = self.normal(scale=1., dim=self.num_asset)
        value = self.smooth(arr=value, window=window, shift=0)
        data = self.add_component(arr=data, other=value, scale=scale)
        return data, value

    def add_condition(self, data: ndarray, factor: ndarray, scale: float) -> Tuple[ndarray, ndarray]:
        """

        """
        # condition signal which picks one factor as conditioner
        value = self.normal(scale=1., dim=self.num_asset)
        value = self.smooth(arr=value, window=self.cond_sig_window, shift=0)
        value_ = value * factor
        data = self.add_component(arr=data, other=value_, scale=scale)
        return data, value

    def add_broken(self, data: ndarray, window: int, scale: float, fraction: float) -> Tuple[ndarray, ndarray, ndarray]:
        """

        """
        # a signal which works for last quarter after start date
        value_ = self.normal(scale=1., dim=self.num_asset)
        value = value_.copy()
        start = (self.num_date - self.date_start) * fraction
        value[-int(start):, :] = -1 * value[-int(start):, :]
        # smooth both the real and broken signal
        value_ = self.smooth(arr=value_, window=window, shift=0)
        value = self.smooth(arr=value, window=window, shift=0)
        data = self.add_component(arr=data, other=value_, scale=scale)
        return data, value, value_

    def create(self, debug: bool, location: str) -> None:
        """

        """
        from numpy import nan
        from numpy.random import rand, shuffle
        float_format = '%.6f'

        x_data, data = self.compute(debug=debug)
        # limit the sample
        x_data = x_data.unstack().shift(-1).iloc[self.date_start:self.date_end].stack()
        y_data = data.iloc[self.date_start:self.date_end].stack()
        y_true = data.iloc[self.date_end]
        # mask % of y_data as missing
        mask = rand(len(y_data)) < self.percent_missing
        y_data[mask] = nan
        # mask x_data signal name
        columns = x_data.columns.copy().to_list()
        shuffle(columns)
        x_data = x_data.reindex(columns=columns)
        rename, index = dict(), 0
        for col in columns:
            if col.startswith('f_'):
                rename[col] = col
            else:
                rename[col] = f's_{index}'
                index += 1
        x_data.rename(columns=rename, inplace=True)
        x_data.sort_index(axis=1, inplace=True)

        if not debug:
            print(rename)
            y_data.unstack().to_csv(f'{location}/returns.csv.gz', compression='gzip', float_format=float_format)
            x_data.to_csv(f'{location}/signals.csv.gz', compression='gzip', float_format=float_format)


def main() -> None:
    """

    """
    from numpy.random import seed
    args = make_parser().parse_args()
    seed(args.seed)
    prob = Problem(signal_scale=args.signal_scale, fake_sig_upscale=args.fake_sig_upscale,
                   fake_sig_fraction=args.fake_sig_fraction, factor_trend_scale=args.factor_trend_scale)
    prob.create(debug=args.debug, location=args.location)


if __name__ == '__main__':
    main()
