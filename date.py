from datetime import date
from pandas import Timestamp
from pandas.tseries.offsets import BaseOffset, DateOffset, Week, MonthBegin, Day
from ..constant import DAY2NS


class Date(date):
    """

    """
    @staticmethod
    def convert(dobj: Timestamp) -> 'Date': return Date(year=dobj.year, month=dobj.month, day=dobj.day)

    @staticmethod
    def from_str(dobj: str) -> 'Date': return Date.convert(Timestamp(dobj))

    @staticmethod
    def from_epoch(epoch: int, tz: str) -> 'Date': return Date.convert(Timestamp(epoch, tz=tz))

    def roll(self, freq: BaseOffset, shift: int) -> 'Date':
        """
            rollback when shift is 0
        """
        if shift == 0:
            ans = freq.rollback(self)
        else:
            ans = self + shift * freq
        return self.convert(dobj=ans)

    def roll_weekly(self, shift: int) -> 'Date':
        """

        """
        freq = Week(weekday=self.weekday())
        return self.roll(freq=freq, shift=shift)

    def roll_monthly(self, shift: int) -> 'Date':
        """

        """
        first = self.replace(day=1).roll(freq=MonthBegin(), shift=shift)
        shift = self.weekday() - first.weekday()
        shift = shift % 7 + self.num_weekday_monthly * 7
        ans = first.roll(freq=Day(), shift=shift)
        return ans

    def roll_annual(self, shift: int) -> 'Date': return self.roll_monthly(shift=shift * 12)

    def end_epoch(self, tz: str) -> int: return Timestamp(self, tz=tz).value + DAY2NS

    def start_epoch(self, tz: str) -> int: return Timestamp(self, tz=tz).value

    @property
    def timestamp(self) -> Timestamp: return Timestamp(self)

    @property
    def num_week_monthly(self) -> int:
        """
            number of week in the month
        """
        first = self.replace(day=1)
        if first.weekday() == 0:
            ans = self.day + 6
        else:
            ans = self.day + first.weekday() - 1
        ans = ans // 7 - 1
        return ans

    @property
    def num_weekday_monthly(self) -> int:
        """
            ans-th weekday of the month
        """
        first = self.replace(day=1)
        ans = self.day + first.weekday() + 6 - self.weekday()
        adj = self.weekday() < first.weekday()
        ans = ans - int(adj) * 7
        ans = ans // 7 - 1
        return ans
