import subprocess
import sys
import re

N = 5

re_duration = re.compile(r'Time \(min:sec\): (\d+:\d+)')  #, re.DOTALL)
re_mem = re.compile(r'Max\. mem \(GiB\): (\d+\.\d+)')
re_n_msgs = re.compile(r'Messages sent: (\d+)')

duration = []
mem = []
n_msgs = []
for i in range(N):
    print('Run %d/%d' % (i+1, N))
    cmd = [sys.executable, 'scenario.py']
    subprocess.check_output(cmd, universal_newlines=True)

    cmd = [sys.executable, 'analysis.py']
    out = subprocess.check_output(cmd, universal_newlines=True)

    m, s = re_duration.search(out).group(1).split(':')
    s = int(m) * 60 + int(s)
    duration.append(s)
    mem.append(float(re_mem.search(out).group(1)))
    n_msgs.append(int(re_n_msgs.search(out).group(1)))

total_t = sum(duration) / N
total_mem = sum(mem) / N
total_msgs = round(sum(n_msgs) / N)

print('%d %.2f %d:%02d' % (total_msgs, total_mem, total_t // 60, total_t % 60))
