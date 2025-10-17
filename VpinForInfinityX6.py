import time
from functools import wraps
from logging import getLevelName, getLogger

import numpy as np
import pandas as pd
import rust_indicators
from freqtrade.strategy import DecimalParameter
from scipy.stats import norm, rankdata

import sys

sys.path.append("/freqtrade")


from NostalgiaForInfinityX6 import NostalgiaForInfinityX6

logger = getLogger(__name__)


def timeit(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    logger.debug(f"{func.__name__} executed in {elapsed:.6f} seconds")
    return result

  return wrapper


@timeit
def create_volume_buckets(df, bucket_size_pct=0.01):
  n_days = df["date"].dt.normalize().nunique()
  avg_daily_volume = df["volume"].sum() / max(1, n_days)
  bucket_size = avg_daily_volume * bucket_size_pct

  df["cum_volume"] = df["volume"].cumsum()
  df["bucket_id"] = (df["cum_volume"] / bucket_size).astype(int)

  buckets = df.groupby("bucket_id").agg({"date": "first", "open": "first", "close": "last", "volume": "sum"})

  buckets.columns = ["datetime", "open", "close", "volume"]
  buckets["price_change"] = buckets["close"] - buckets["open"]
  buckets = buckets.reset_index()

  return buckets, bucket_size


@timeit
def bulk_volume_classification(buckets, window=50):
  buckets["ret"] = np.log(buckets["close"]).diff().fillna(0.0)
  buckets["sigma"] = buckets["ret"].rolling(window, min_periods=10).std()
  buckets["sigma"] = buckets["sigma"].bfill()
  buckets["sigma"] = buckets["sigma"].replace(0, np.nan).fillna(buckets["ret"].std())

  sigma_floor = 1e-6
  buckets["z_score"] = buckets["ret"] / (buckets["sigma"].clip(lower=sigma_floor))
  buckets["buy_prob"] = norm.cdf(buckets["z_score"])
  buckets["buy_volume"] = buckets["volume"] * buckets["buy_prob"]
  buckets["sell_volume"] = buckets["volume"] * (1 - buckets["buy_prob"])

  return buckets


@timeit
def calculate_vpin_raw(df, ta, bucket_size_pct=0.01, vpin_window=50):
  buckets, bucket_size = create_volume_buckets(df, bucket_size_pct)
  buckets = bulk_volume_classification(buckets, window=vpin_window)

  if len(buckets) <= 2:
    raise ValueError("Too few buckets. Reduce bucket_size_pct.")
  if vpin_window >= len(buckets):
    vpin_window = max(2, int(0.2 * len(buckets)))

  vpin_values = np.array(ta.vpin(buckets["buy_volume"].values, buckets["sell_volume"].values, vpin_window))

  buckets["vpin"] = vpin_values

  df["bucket_id"] = (df["volume"].cumsum() / bucket_size).astype(int)
  vpin_map = dict(zip(buckets["bucket_id"], vpin_values, strict=True))
  df["vpin"] = df["bucket_id"].map(vpin_map)

  return df, buckets, vpin_values


@timeit
def calculate_vpin_cdf(vpin_values):
  out = np.full_like(vpin_values, np.nan, dtype=float)
  m = ~np.isnan(vpin_values)
  if m.sum() == 0:
    return out
  ranks = rankdata(vpin_values[m], method="average")
  out[m] = ranks / (len(ranks) + 1.0)
  return out


@timeit
def interpolate_to_timeframe(buckets, df, bucket_name):
  bucket_series = pd.Series(buckets[bucket_name].values, index=pd.to_datetime(buckets["datetime"]))

  is_bool = pd.api.types.is_bool_dtype(bucket_series)
  if is_bool:
    bucket_series = bucket_series.astype(int)

  aligned = bucket_series.reindex(df["date"])
  aligned = aligned.interpolate(method="linear", limit_direction="both", limit_area="inside")
  aligned = aligned.bfill().ffill()

  if is_bool:
    return (aligned > 0.5).astype(bool)
  return aligned


class VpinForInfinityX6(NostalgiaForInfinityX6):
  vpin_treshold = DecimalParameter(0.85, 0.99, decimals=3, default=0.9, space="buy")

  def __init__(self, config: dict) -> None:
    super().__init__(config)
    self._ta = None

  def __getstate__(self):
    state = self.__dict__.copy()
    state["_ta"] = None
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._ta = None

  @property
  def ta(self):
    if self._ta is None:
      self._ta = rust_indicators.RustTA()
      logger.info(f"✓ Rust Indicators Backend: {self._ta.device()}")
    return self._ta

  def version(self) -> str:
    return super().version() + "-vpin"

  @property
  def plot_config(self):
    existing_config = super().plot_config
    # existing_config["subplots"]["vpin_raw"] = {"vpin_raw": {"color": "blue"}}
    existing_config["subplots"]["VPIN CDF"] = {"vpin_cdf_check": {"color": "red"}, "vpin_cdf_ssf": {"color": "cyan"}}
    existing_config["subplots"]["Protections Superclass"] = {"protections_long_global": {"color": "yellow"}}
    existing_config["subplots"].pop("long_pump_protection", None)
    existing_config["subplots"].pop("long_dump_protection", None)
    return existing_config

  def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    df = super().populate_indicators(df, metadata)

    logger.debug("Calculating academic VPIN...")

    if ("trading_mode" in self.config) and (self.config["trading_mode"] in ["futures", "margin"]):
      btc_info_pair = f"BTC/{self.config['stake_currency']}:{self.config['stake_currency']}"
    else:
      btc_info_pair = f"BTC/{self.config['stake_currency']}"

    btc_informative = self.btc_info_switcher(btc_info_pair, self.timeframe, metadata)
    btc_informative["open"] = btc_informative["btc_open"]
    btc_informative["high"] = btc_informative["btc_high"]
    btc_informative["low"] = btc_informative["btc_low"]
    btc_informative["close"] = btc_informative["btc_close"]
    btc_informative["volume"] = btc_informative["btc_volume"]

    try:
      df_temp, buckets, vpin_raw = calculate_vpin_raw(btc_informative, self.ta, bucket_size_pct=0.01)
    except ValueError as e:
      logger.warning(f"VPIN calculation failed: {e}")
      return df

    if len(buckets) < 4:
      logger.warning("Insufficient buckets for VPIN calculation")
      return df

    vpin_cdf = calculate_vpin_cdf(vpin_raw)
    buckets["vpin_cdf"] = vpin_cdf

    buckets["vpin_cdf_ssf"] = self.ta.supersmoother(buckets["vpin_cdf"].values, period=20)
    buckets["vpin_cdf_check"] = buckets["vpin_cdf_ssf"] <= self.vpin_treshold.value

    # df["vpin_raw"] = interpolate_to_timeframe(vpin_raw, df).values
    df["vpin_cdf"] = interpolate_to_timeframe(buckets, df, "vpin_cdf").values
    df["vpin_cdf_ssf"] = interpolate_to_timeframe(buckets, df, "vpin_cdf_ssf").values
    df["vpin_cdf_check"] = interpolate_to_timeframe(buckets, df, "vpin_cdf_check").values

    if "protections_long_global" in df.columns:
      df.loc[~df["vpin_cdf_check"], "protections_long_global"] = False
    else:
      logger.warning("protections_long_global not found in parent DataFrame")

    if len(buckets) >= 4:
      recent_ok = buckets["vpin_cdf_check"].iloc[-4:-1].all()
      current_toxic = not buckets["vpin_cdf_check"].iloc[-1]

      if recent_ok and current_toxic:
        vpin_cdf_ssf = buckets["vpin_cdf_ssf"].iloc[-1]
        msg = f"⚠️ Extreme Toxicity Probability: {int(vpin_cdf_ssf * 100)}% > {int(self.vpin_treshold.value * 100)}%"
        self.dp.send_msg(msg)

    if getLevelName(logger.getEffectiveLevel()) == "DEBUG":
      alignment_check = pd.DataFrame(
        {
          "datetime": df["date"].tail(20),
          "vpin_safe": df["vpin_cdf_check"].tail(20),
          "protection": df["protections_long_global"].tail(20),
        }
      )
      logger.debug(f"\nAlignment check:\n{alignment_check}")

      missing_pct = df["vpin_cdf_check"].isna().mean() * 100
      logger.debug(f"Missing after interpolation: {missing_pct:.2f}%")

      logger.info(f"Buckets: {len(buckets)} volume-synchronized buckets")
      logger.info(f"VPIN protection applied to {len(df)} candles")
      logger.info(f"Toxic periods: {(~df['vpin_cdf_check']).sum()} candles")

    return df
