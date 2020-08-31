expected=0.49748743718592964
_RANGE = (-15, 51)

N = len(range(*_RANGE))
p = 1./N

tot = 0.
for y in range(*_RANGE):
    for n in range(*_RANGE):
        tot += (p-float((n-y)>=0))**2

print(tot/(N*N))
