from collections import defaultdict
ideas = ["coffee","donuts","time","toffee"]
groups = defaultdict(set)

print(groups)

groups = defaultdict(set)

for s in ideas:
    groups[s[0]].add(s[1:])  # 按照首字母分组


print(groups)

