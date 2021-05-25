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

#Split Training, Test and Holdout
import json

Training = []
Test = []
Holdout = []

splitFile = open("shuffled.txt","r")

for lines in splitFile:
    line = lines.rstrip().split("\t")
    sample = line[0]
    Set = line[1]
    if Set == "Training":
        Training.append(sample)
    if Set == "Test":
        Test.append(sample)
    if Set == "Holdout":
        Holdout.append(sample)
    else:
        continue

with open("Features.txt") as json_file:
    Features = json.load(json_file)

FeaturesTest = {}
FeaturesHoldout = {}
FeaturesTraining = {}

for sample in Features.keys():
    if sample in Training:
        FeaturesTraining[sample] = Features[sample]
    elif sample in Test:
        FeaturesTest[sample] = Features[sample]
    elif sample in Holdout:
        FeaturesHoldout[sample] = Features[sample]

with open("FeaturesTest.txt", "w") as outfile:
    json.dump(FeaturesTest, outfile)

with open("FeaturesHoldout.txt", "w") as outfile:
    json.dump(FeaturesHoldout, outfile)

with open("FeaturesTraining.txt", "w") as outfile:
    json.dump(FeaturesTraining, outfile)