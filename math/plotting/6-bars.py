#!/usr/bin/env python3
"""Module that plots a stacked bar graph of fruit per person."""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plot the quantity of each fruit owned by each person."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    fruits = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    bottom = np.zeros(3)
    for row, name, color in zip(fruit, fruits, colors):
        plt.bar(people, row, width=0.5, bottom=bottom,
                color=color, label=name)
        bottom += row
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()
    plt.show()
