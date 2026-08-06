# 0x01. Plotting

## Description

This project covers data visualization with `matplotlib`, applied to numerical/statistical data as preparation for machine learning work. It goes through line graphs, scatter plots, logarithmic scaling, multi-line plots with legends, histograms, multi-plot figures, and stacked bar charts.

## Requirements

- Ubuntu 20.04 LTS
- Python 3.9
- NumPy 1.25.2
- Matplotlib 3.8.3
- pycodestyle 2.11.1
- All files must be executable
- All files must end with a new line
- The first line of all files must be exactly `#!/usr/bin/env python3`
- All modules and functions must have a documentation string

## Files

| File | Description |
|---|---|
| `0-line.py` | `line()` — plot `y = x³` as a solid red line, x-axis from 0 to 10 |
| `1-scatter.py` | `scatter()` — scatter plot of men's height vs weight, magenta points |
| `2-change_scale.py` | `change_scale()` — line graph of C-14 decay, y-axis log-scaled |
| `3-two.py` | `two()` — C-14 (dashed red) and Ra-226 (solid green) decay curves with legend |
| `4-frequency.py` | `frequency()` — histogram of student grades, bins every 10, black-outlined bars |
| `5-all_in_one.py` | `all_in_one()` — the 5 previous plots combined into one 3x2 figure |
| `6-bars.py` | `bars()` — stacked bar chart of fruit counts per person |

## Usage

Each task is testable via its associated `X-main.py` file:

```bash
./0-main.py
```

## Author

Paul — Holberton School
