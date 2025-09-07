import math
import random
import subprocess, sys
#numbers = [70000,80000,90000,100000]
numbers = [100,500,1000]
change_seeds = False
seed = 23945820
for i in range(0, len(numbers)):
    if change_seeds:
        seed = random.randint(1,10000000000)
    subprocess.run([sys.executable, "AllInOne.py", str(numbers[i]), str(seed)])
