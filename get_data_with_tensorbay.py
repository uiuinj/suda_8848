from tensorbay import GAS
from tensorbay.dataset import Dataset
import os
os.chdir("test")
gas = GAS("Accesskey-16d5f0f7386759dfec202786fd269ccc")

dataset = Dataset("RP2K", gas)
print(dataset.keys())
segment = dataset["test"]
a = segment[0]
li = segment[0:2000]
for i in li:
    name = i.path
    fp = i.open()
    a = fp.read()
    classification_category = i.label.classification.category
    print(classification_category)
#     print()
    # os.path.exists(path) 判断一个目录是否存在
    if os.path.exists(classification_category):
        with open(classification_category+"/"+name,"wb") as f:
            f.write(a)
    else:
        os.mkdir(classification_category)
        with open(classification_category + "/"+name, "wb") as f:
            f.write(a)
print('finish')