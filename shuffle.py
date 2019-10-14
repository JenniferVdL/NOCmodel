from random import shuffle
from math import ceil

nocs = {}
with open("nocs.txt") as f:
  for line in f:
    sample, noc = line.rstrip().split("\t")
    if noc not in nocs:
      nocs[noc] = []
    nocs[noc].append(sample)


with open("shuffled.txt", "w") as f:
    for noc in nocs:
      samples = nocs[noc]
      shuffle(samples)
      n = len(samples)
      tv = int(ceil(n/5.0))
      for i in range(n):
        f.write("\t".join((samples[i], "Holdout" if i < tv else "Test" if i < 2*tv else "Train", noc)) + "\n")

