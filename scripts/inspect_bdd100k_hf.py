"""Inspect BDD100K sample attributes and labels."""

import json
import requests

url = "https://huggingface.co/datasets/dgural/bdd100k/resolve/main/samples.json"
print("Fetching byte range from samples.json...")
r = requests.get(url, headers={"Range": "bytes=0-120000"})
text = r.text

# Find individual sample objects
# Each sample has a "filepath": "data/..."
import re
pattern = r'\{"_id": \{"\$oid": "[a-f0-9]+"\}[^}]*?"filepath": "[^"]+"'
matches = list(re.finditer(pattern, text))
# Find the first sample end
second_start = matches[1].start()
sample0 = text[:second_start]
print("Keys/structure in sample 0:")
print("Attributes snippet:")
print(sample0[10850:11250])
