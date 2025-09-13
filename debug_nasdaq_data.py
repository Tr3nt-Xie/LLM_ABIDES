#!/usr/bin/env python3
"""
Debug why NASDAQ data isn't showing
"""

import sys
sys.path.insert(0, 'src')
from itch_data_parser import LOBSTERDataParser
import numpy as np

print("Loading real NASDAQ data...")
parser = LOBSTERDataParser("AMZN", "2012-06-21", data_dir="/workspace")
parser.parse_messages()
parser.parse_orderbook()

real_mid_prices = []
real_timestamps = []
for snapshot in parser.orderbook_snapshots:
    if snapshot.bid_price > 0 and snapshot.ask_price > 0:
        mid_price = (snapshot.bid_price + snapshot.ask_price) / 2
        real_mid_prices.append(mid_price)
        real_timestamps.append(snapshot.timestamp)

print(f"\nNumber of NASDAQ data points: {len(real_mid_prices)}")
if len(real_mid_prices) > 0:
    print(f"Time range: {real_timestamps[0]/3600:.2f} - {real_timestamps[-1]/3600:.2f} hours")
    print(f"Price range: ${real_mid_prices[0]:.2f} - ${real_mid_prices[-1]:.2f}")
    print(f"Min price: ${np.min(real_mid_prices):.2f}")
    print(f"Max price: ${np.max(real_mid_prices):.2f}")
    print(f"First few timestamps (hours): {[t/3600 for t in real_timestamps[:5]]}")
else:
    print("ERROR: No NASDAQ data loaded!")

# Check if files exist
import os
files = [
    "AMZN_2012-06-21_34200000_57600000_message_1.csv",
    "AMZN_2012-06-21_34200000_57600000_orderbook_1.csv"
]
for f in files:
    exists = os.path.exists(f"/workspace/{f}")
    print(f"\n{f}: {'EXISTS' if exists else 'MISSING'}")