import time
from functools import wraps
from logging import getLogger

import pandas as pd
import numpy as np
from freqtrade.strategy import DecimalParameter, IntParameter
import sys
from scipy.stats import rankdata

sys.path.append("/freqtrade")
from NostalgiaForInfinityX7 import NostalgiaForInfinityX7

import warnings

# Cython modules
from vpin_fast import calculate_academic_vpin, USE_CYTHON as USE_CYTHON_VPIN
from ehlers import supersmoother_cython, USE_CYTHON as USE_CYTHON_EHLERS

logger = getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning, module="freqtrade.optimize.backtesting")


if not USE_CYTHON_VPIN or not USE_CYTHON_EHLERS: # or not USE_CYTHON_MML:
  raise ImportError("Cython modules required. Run: pip install -e .")


def timeit(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    logger.debug(f"{func.__name__}: {elapsed:.6f}s")
    return result

  return wrapper


def calculate_vpin_cdf(vpin_values):
  """Rank-based CDF transformation"""

  out = np.full_like(vpin_values, np.nan, dtype=np.float64)
  mask = ~np.isnan(vpin_values)

  if mask.sum() == 0:
    return out

  ranks = rankdata(vpin_values[mask], method="average")
  out[mask] = ranks / (len(ranks) + 1.0)
  return out


def interpolate_to_timeframe(buckets, df, bucket_name):
  """Linear interpolation for irregular buckets"""
  bucket_series = pd.Series(buckets[bucket_name].values, index=pd.to_datetime(buckets["datetime"]))

  is_bool = pd.api.types.is_bool_dtype(bucket_series)
  if is_bool:
    bucket_series = bucket_series.astype(int)

  aligned = bucket_series.reindex(df["date"])
  aligned = aligned.interpolate(method="linear", limit_direction="both", limit_area="inside")
  aligned = aligned.bfill().ffill()

  if is_bool:
    return (aligned > 0.5).astype(bool).values
  return aligned.values


class VpinGuardX7(NostalgiaForInfinityX7):
  """
  Order flow toxicity protection using VPIN (Volume-Synchronized
  Probability of Informed Trading) with Cython acceleration.
  """

  vpin_treshold = DecimalParameter(0.85, 0.99, decimals=3, default=0.9, space="sell")
  vpin_treshold = DecimalParameter(0.85, 0.99, decimals=3, default=0.9, space="sell")
  vpin_ssf = IntParameter(2, 30, default=20, space="sell")

  bucket_size_pct = DecimalParameter(0.001, 0.3, decimals=3, default=0.01, space="sell")
  vpin_window = IntParameter(5, 200, default=50, space="sell")

  def version(self) -> str:
    return super().version() + "-vpin-cython"

  @property
  def plot_config(self):
    existing_config = super().plot_config
    existing_config["subplots"]["VPIN CDF"] = {
      "vpin_raw": {"color": "blue"},
      "vpin_cdf": {"color": "#ff5bdb"},
    }
    existing_config["subplots"]["Protections Superclass"] = {
      "protections_long_global": {"color": "yellow"},
      "vpin_cdf_check": {"color": "red"},
      "vpin_cdf_check_coin": {"color": "#ffaf00"},
    }
    existing_config["subplots"].pop("long_pump_protection", None)
    existing_config["subplots"].pop("long_dump_protection", None)
    return existing_config

  @timeit
  def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    df = super().populate_indicators(df, metadata)
    pair = metadata.get("pair")

    logger.debug("Calculating VPIN (Cython)...")

    # Get BTC informative data
    # if self.config.get("trading_mode") in ["futures", "margin"]:
    #   btc_pair = f"BTC/{self.config['stake_currency']}:{self.config['stake_currency']}"
    # else:
    #   btc_pair = f"BTC/{self.config['stake_currency']}"
    #
    # btc_inf = self.btc_info_switcher(btc_pair, self.timeframe, metadata)
    # btc_inf["open"] = btc_inf["btc_open"]
    # btc_inf["close"] = btc_inf["btc_close"]
    # btc_inf["volume"] = btc_inf["btc_volume"]
    #
    # btc_inf["datetime"] = btc_inf["date"]

    coin_inf = df.copy()
    coin_inf["datetime"] = coin_inf["date"]

    # Calculate VPIN
    try:
      df_temp, buckets, vpin_raw = calculate_academic_vpin(
        coin_inf, bucket_size_pct=self.bucket_size_pct.value, vpin_window=self.vpin_window.value
      )
    except ValueError as e:
      logger.warning(f"VPIN failed: {e}")
      return df

    if len(buckets) < 4:
      logger.warning("Insufficient buckets")
      return df

    # Transform to CDF and smooth (Cython supersmoother)
    vpin_cdf = calculate_vpin_cdf(vpin_raw)
    buckets["vpin_cdf"] = vpin_cdf
    buckets["vpin_cdf_ssf"] = supersmoother_cython(vpin_cdf, period=self.vpin_ssf.value)
    buckets["vpin_cdf_check"] = buckets["vpin_cdf_ssf"] <= self.vpin_treshold.value

    # Interpolate to strategy timeframe
    buckets["vpin_raw"] = vpin_raw

    df["vpin_raw"] = interpolate_to_timeframe(buckets, df, "vpin_raw")
    df["vpin_cdf"] = interpolate_to_timeframe(buckets, df, "vpin_cdf")
    df["vpin_cdf_ssf"] = interpolate_to_timeframe(buckets, df, "vpin_cdf_ssf")
    df["vpin_cdf_check"] = interpolate_to_timeframe(buckets, df, "vpin_cdf_check")

    # Apply protection filter
    df.loc[~df["vpin_cdf_check"], "protections_long_global"] = False

    if len(buckets) >= 2:
      curr_ok = buckets["vpin_cdf_check"].iloc[-1]
      prev_ok = buckets["vpin_cdf_check"].iloc[-2]
      ssf_val = buckets["vpin_cdf_ssf"].iloc[-1]

      recent_toxic = (not curr_ok) and prev_ok  # Flipped from safe to toxic
      recent_cleared = curr_ok and (not prev_ok)  # Flipped from toxic to safe

      if recent_toxic:
        msg = f"⚠️  {pair} Extreme Toxicity: {round(ssf_val * 100, 2)}% > {round(self.vpin_treshold.value * 100, 2)}%"
        self.dp.send_msg(msg)
      elif recent_cleared:
        msg = f"✅ {pair} Toxicity Cleared: {round(ssf_val * 100, 2)}% ≤ {round(self.vpin_treshold.value * 100, 2)}%"
        self.dp.send_msg(msg)

    # Debug logging
    if logger.isEnabledFor(10):  # DEBUG level
      toxic_pct = (~df["vpin_cdf_check"]).sum() / len(df) * 100
      logger.debug(
        f"Buckets: {len(buckets)} | Toxic: {toxic_pct:.1f}% | Missing: {df['vpin_cdf_check'].isna().mean() * 100:.1f}%"
      )

    return df

