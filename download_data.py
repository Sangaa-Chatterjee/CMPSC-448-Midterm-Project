"""Download the instructor-suggested public TXD-22 archive; no authentication bypass."""
from pathlib import Path
from urllib.request import urlretrieve
p=Path('data/txd22.zip');p.parent.mkdir(exist_ok=True)
urlretrieve('https://www.kaggle.com/api/v1/datasets/download/mdsadiqiqbal/txd-22-a-large-scale-benchmark-dataset',p)
print(p)
