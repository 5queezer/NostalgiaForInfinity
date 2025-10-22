import time
from functools import wraps
from logging import getLogger

import pandas as pd
import numpy as np
from freqtrade.strategy import DecimalParameter, IntParameter
import sys

sys.path.append("/freqtrade")
from NostalgiaForInfinityX6 import NostalgiaForInfinityX6

# Cython modules
from vpin_fast import vpin_cython, bvc_classify, supersmoother_cython, USE_CYTHON as USE_CYTHON_VPIN
from murrey_math_levels import compute_levels, USE_CYTHON as USE_CYTHON_MML

logger = getLogger(__name__)

if not USE_CYTHON_VPIN or not USE_CYTHON_MML:
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


@timeit
def calculate_vpin_raw(df, bucket_size_pct=0.01, vpin_window=50):
  """Cython-accelerated VPIN pipeline"""

  # Step 1: Create volume buckets
  n_days = df["date"].dt.normalize().nunique()
  avg_daily_volume = df["volume"].sum() / max(1, n_days)
  bucket_size = avg_daily_volume * bucket_size_pct

  cum_vol = np.cumsum(df["volume"].values)
  bucket_ids = (cum_vol / bucket_size).astype(np.int64)
  df = df.copy()
  df["bucket_id"] = bucket_ids

  # Step 2: Aggregate to buckets
  buckets = df.groupby("bucket_id", as_index=False).agg(
    {"date": "first", "open": "first", "close": "last", "volume": "sum"}
  )
  buckets.columns = ["bucket_id", "datetime", "open", "close", "volume"]

  # Step 3: Classify buy/sell volumes (Cython)
  buy_vols, sell_vols = bvc_classify(
    buckets["close"].values,
    buckets["volume"].values,
    vpin_window,
  )

  # Step 4: Calculate VPIN (Cython)
  vpin_vals = vpin_cython(buy_vols, sell_vols, vpin_window)
  buckets["vpin"] = vpin_vals

  # Step 5: Map back to original timeframe
  df = df.merge(buckets[["bucket_id", "vpin"]], on="bucket_id", how="left")

  return df, buckets, vpin_vals


@timeit
def calculate_vpin_cdf(vpin_values):
  """Rank-based CDF transformation"""
  from scipy.stats import rankdata

  out = np.full_like(vpin_values, np.nan, dtype=np.float64)
  mask = ~np.isnan(vpin_values)

  if mask.sum() == 0:
    return out

  ranks = rankdata(vpin_values[mask], method="average")
  out[mask] = ranks / (len(ranks) + 1.0)
  return out


@timeit
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


class VpinGuardX6(NostalgiaForInfinityX6):
  """
  Order flow toxicity protection using VPIN (Volume-Synchronized
  Probability of Informed Trading) with Cython acceleration.
  """

  vpin_treshold = DecimalParameter(0.85, 0.99, decimals=3, default=0.933, space="sell")
  vpin_ssf = IntParameter(2, 30, default=11, space="sell")

  def version(self) -> str:
    return super().version() + "-vpin-cython"

  @property
  def plot_config(self):
    existing_config = super().plot_config
    existing_config["main_plot"].update(
      {
        "[+2/8]P": {"color": "#ff6b6b"},
        "[8/8]P": {"color": "#ff8787"},
        "[4/8]P": {"color": "#ffd43b"},
        "[0/8]P": {"color": "#51cf66"},
        "[-2/8]P": {"color": "#339af0"},
      }
    )

    existing_config["subplots"]["VPIN CDF"] = {
      "vpin_raw": {"color": "blue"},
      "vpin_cdf": {"color": "red"},
      "vpin_cdf_ssf": {"color": "cyan"},
    }
    existing_config["subplots"]["Protections Superclass"] = {
      "protections_long_global": {"color": "yellow"},
      "vpin_cdf_check": {"color": "red"},
    }
    existing_config["subplots"]["Murray Math"] = {
      "mmlextreme_oscillator": {"color": "green"},
      "mmlextreme_oscillator_ssf": {"color": "blue"},
    }
    existing_config["subplots"].pop("long_pump_protection", None)
    existing_config["subplots"].pop("long_dump_protection", None)
    return existing_config

  def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    df = super().populate_indicators(df, metadata)

    logger.debug("Calculating VPIN (Cython)...")

    # Get BTC informative data
    if self.config.get("trading_mode") in ["futures", "margin"]:
      btc_pair = f"BTC/{self.config['stake_currency']}:{self.config['stake_currency']}"
    else:
      btc_pair = f"BTC/{self.config['stake_currency']}"

    btc_inf = self.btc_info_switcher(btc_pair, self.timeframe, metadata)
    btc_inf["open"] = btc_inf["btc_open"]
    btc_inf["close"] = btc_inf["btc_close"]
    btc_inf["volume"] = btc_inf["btc_volume"]

    # Calculate VPIN
    try:
      df_temp, buckets, vpin_raw = calculate_vpin_raw(btc_inf, bucket_size_pct=0.01, vpin_window=50)
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

    # levels = compute_levels(df["high"].values, df["low"].values, window=5)
    # df = pd.concat([df, levels], axis=1)
    # df["mmlextreme_oscillator"] = 100 * (df["close"] - df["[-3/8]P"]) / (df["[+3/8]P"] - df["[-3/8]P"])
    # df["mmlextreme_oscillator_ssf"] = supersmoother_cython(df["mmlextreme_oscillator"].values, period=5)

    # Alert on extreme toxicity
    if len(buckets) >= 4:
      recent_toxic = buckets["vpin_cdf_check"].iloc[-4:-1].any()
      current_toxic = buckets["vpin_cdf_check"].iloc[-1]
      ssf_val = buckets["vpin_cdf_ssf"].iloc[-1]

      if not recent_toxic and current_toxic:
        msg = f"⚠️ Extreme Toxicity: {int(ssf_val * 100)}% > {int(self.vpin_treshold.value * 100)}%"
        self.dp.send_msg(msg)
      elif recent_toxic and not current_toxic:
        msg = f"✅ Toxicity Cleared: {int(ssf_val * 100)}% ≤ {int(self.vpin_treshold.value * 100)}%"
        self.dp.send_msg(msg)

    # Debug logging
    if logger.isEnabledFor(10):  # DEBUG level
      toxic_pct = (~df["vpin_cdf_check"]).sum() / len(df) * 100
      logger.debug(
        f"Buckets: {len(buckets)} | Toxic: {toxic_pct:.1f}% | Missing: {df['vpin_cdf_check'].isna().mean() * 100:.1f}%"
      )

    return df
