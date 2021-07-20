import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

# https://seaborn.pydata.org/tutorial/function_overview.html
penguins = sns.load_dataset("penguins")

print(penguins.head(10))
sns.histplot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")
plt.show()

sns.kdeplot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")
plt.show()

# displot with kind
sns.displot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")  # kind=hist
sns.displot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack", kind="kde")
plt.show()

sns.displot(data=penguins, x="flipper_length_mm", hue="species", col="species")
plt.show()


f, axs = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw=dict(width_ratios=[4, 3]))
sns.scatterplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species", ax=axs[0])
sns.histplot(data=penguins, x="species", hue="species", shrink=.8, alpha=.8, legend=False, ax=axs[1])
f.tight_layout()
plt.show()

# Figure-level functions own their figure
tips = sns.load_dataset("tips")
g = sns.relplot(data=tips, x="total_bill", y="tip")
g.ax.axline(xy1=(10, 2), slope=.2, color="b", dashes=(5, 2))
plt.show()

# Customizing plots from a figure-level function
g = sns.relplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", col="sex")
g.set_axis_labels("Flipper length (mm)", "Bill length (mm)")
plt.show()

# subplots
f, ax = plt.subplots()
f, ax = plt.subplots(1,2, sharey=True)
sns.FacetGrid(penguins)
sns.FacetGrid(penguins, col="sex")
sns.FacetGrid(penguins, col="sex", height=3.5, aspect=.75)
plt.show()

# Combining multiple views on the data
sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species")
plt.show()


sns.pairplot(data=penguins, hue="species")
plt.show()

sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species", kind="hist")
plt.show()

