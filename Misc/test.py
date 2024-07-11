import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

start_m =100
wager = 5
bets = 5
trials = 1000

trans = np.vectorize(lambda t: -wager if t <=0.51 else wager)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1,1,1)

end_m = []

for i in range(trials):
    money = reduce(lambda c, x: c + [c[-1] + x], trans(np.random.random(bets)), [start_m])
    end_m.append(money[-1])
    plt.plot(money)

plt.ylabel('Player Money in $')
plt.xlabel('Number of bets')
plt.title(f"John starts the game with $ {start_m:.2f} and ends with $ {sum(end_m) / len(end_m):.2f}")
plt.show()