import time
import requests
import planetary_computer as pc
import pystac_client

client = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
s1 = list(client.search(collections=["sentinel-1-grd"], bbox=[76.55, 12.25, 76.70, 12.35], datetime="2026-08-18/2026-09-17", limit=1).items())[0]
href = pc.sign(s1.assets["vv"].href)

t0 = time.time()
with requests.get(href, stream=True) as r:
    r.raise_for_status()
    total = 0
    with open("D:\\satQai\\data\\real_test\\speed_test.tmp", "wb") as f:
        for chunk in r.iter_content(chunk_size=4*1024*1024):
            if chunk:
                f.write(chunk)
                total += len(chunk)
                if total >= 30 * 1024 * 1024:
                    break
t1 = time.time()
mb = total / (1024 * 1024)
duration = t1 - t0
print(f"Downloaded {mb:.1f} MB in {duration:.2f}s ({(mb/duration):.2f} MB/s)")
