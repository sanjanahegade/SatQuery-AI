import sys
from pystac_client import Client
import planetary_computer as pc

endpoint = "https://planetarycomputer.microsoft.com/api/stac/v1"
client = Client.open(endpoint, modifier=pc.sign_inplace)

bbox = [76.55, 12.25, 76.70, 12.35]
datetime_range = "2026-08-18/2026-09-17"

print("Querying Sentinel-2 with bbox and datetime:", bbox, datetime_range)
try:
    s2_search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=datetime_range,
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        limit=5
    )
    s2_items = list(s2_search.items())
    print(f"Sentinel-2 items found: {len(s2_items)}")
    for item in s2_items:
        print(f"  ID: {item.id}, date: {item.datetime}, cloud: {item.properties.get('eo:cloud_cover')}")
        print("  Available assets:", list(item.assets.keys())[:10])
except Exception as e:
    print(f"Error querying Sentinel-2: {e}")

print("\nQuerying Sentinel-1 with bbox and datetime:", bbox, datetime_range)
try:
    s1_search = client.search(
        collections=["sentinel-1-grd"],
        bbox=bbox,
        datetime=datetime_range,
        limit=5
    )
    s1_items = list(s1_search.items())
    print(f"Sentinel-1 items found: {len(s1_items)}")
    for item in s1_items:
        mode = (item.properties.get("sar:instrument_mode") or 
                item.properties.get("sar:mode") or 
                item.properties.get("instrument_mode"))
        print(f"  ID: {item.id}, date: {item.datetime}, mode: {mode}")
        print("  Available assets:", list(item.assets.keys()))
        print("  Polarizations:", item.properties.get("sar:polarizations"))
except Exception as e:
    print(f"Error querying Sentinel-1: {e}")
