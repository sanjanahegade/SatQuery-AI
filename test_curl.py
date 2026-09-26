import subprocess
import time
import planetary_computer as pc
import pystac_client

client = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
bbox = [76.55, 12.25, 76.70, 12.35]
s1 = list(client.search(collections=["sentinel-1-grd"], bbox=bbox, datetime="2026-08-18/2026-09-17", limit=1).items())[0]
vv_url = pc.sign(s1.assets["vv"].href)

print("Testing curl with signed URL...")
t0 = time.time()
# Download 20MB using Range header with curl
res = subprocess.run([
    "curl.exe", "-s", "-r", "0-20971520",
    "-o", "D:\\satQai\\data\\real_test\\test_chunk.dat",
    vv_url
], capture_output=True)
t1 = time.time()
print(f"Curl downloaded 20MB in {t1-t0:.2f}s ({(20/(t1-t0)):.2f} MB/s)")
