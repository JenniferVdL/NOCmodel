#
# Copyright (C) 2021, Netherlands Forensic Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

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

