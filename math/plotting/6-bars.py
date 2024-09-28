#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

people = ['Farrah', 'Fred', 'Felicia']


# Plot the stacked bar chart
plt.bar(people, fruit[0], color='red', width=0.5, label='apples')
plt.bar(people, fruit[1], bottom=fruit[0], color='yellow', width=0.5, label='bananas')
plt.bar(people, fruit[2], bottom=fruit[0] + fruit[1], color='#ff8000', width=0.5, label='oranges')
plt.bar(people, fruit[3], bottom=fruit[0] + fruit[1] + fruit[2], color='#ffe5b4', width=0.5, label='peaches')

# Add labels and title
plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80)
plt.title('Number of Fruit per Person')

# Add legend
plt.legend()

# Display the plot
plt.show()
