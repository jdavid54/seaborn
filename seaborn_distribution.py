import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

# https://seaborn.pydata.org/tutorial/distributions.html

penguins = sns.load_dataset("penguins")
sns.displot(penguins, x="flipper_length_mm")
plt.show()

sns.displot(penguins, x="flipper_length_mm", binwidth=3)
sns.displot(penguins, x="flipper_length_mm", bins=20)
plt.show()

tips = sns.load_dataset("tips")
sns.displot(tips, x="size")
plt.show()

sns.displot(tips, x="size", bins=[1, 2, 3, 4, 5, 6, 7])
sns.displot(tips, x="size", discrete=True)
plt.show()

sns.displot(tips, x="day", shrink=.8)
plt.show()

# Conditioning on other variables
sns.displot(penguins, x="flipper_length_mm", hue="species")
plt.show()

sns.displot(penguins, x="flipper_length_mm", hue="species", element="step")
plt.show()

sns.displot(penguins, x="flipper_length_mm", hue="species", multiple="stack")
plt.show()

sns.displot(penguins, x="flipper_length_mm", hue="sex", multiple="dodge")
plt.show()

sns.displot(penguins, x="flipper_length_mm", col="sex", multiple="dodge")
plt.show()

# Normalized histogram statistics
sns.displot(penguins, x="flipper_length_mm", hue="species", stat="density")
plt.show()

sns.displot(penguins, x="flipper_length_mm", hue="species", stat="density", common_norm=False)
plt.show()

sns.displot(penguins, x="flipper_length_mm", hue="species", stat="probability")
plt.show()

# Kernel density estimation
sns.displot(penguins, x="flipper_length_mm", kind="kde")
plt.show()

# Choosing the smoothing bandwidth
sns.displot(penguins, x="flipper_length_mm", kind="kde", bw_adjust=.25)
sns.displot(penguins, x="flipper_length_mm", kind="kde", bw_adjust=2)
plt.show()

# Conditioning on other variables
sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde")
plt.show()

sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde", multiple="stack")
plt.show()

sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde", fill=True)
plt.show()

# Kernel density estimation pitfalls
sns.displot(tips, x="total_bill", kind="kde")
plt.show()

sns.displot(tips, x="total_bill", kind="kde", cut=0)
plt.show()

diamonds = sns.load_dataset("diamonds")
sns.displot(diamonds, x="carat", kind="kde")
plt.show()

sns.displot(diamonds, x="carat")
plt.show()

sns.displot(diamonds, x="carat", kde=True)
plt.show()

# Empirical cumulative distributions
sns.displot(penguins, x="flipper_length_mm", kind="ecdf")
plt.show()

sns.displot(penguins, x="flipper_length_mm", hue="species", kind="ecdf")
plt.show()

# Visualizing bivariate distributions
sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm")
plt.show()

sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind="kde")
plt.show()

sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", hue="species")
plt.show()

sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", hue="species", kind="kde")
plt.show()

sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", binwidth=(2, .5))
plt.show()

sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", binwidth=(2, .5), cbar=True)
plt.show()

sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind="kde", thresh=.2, levels=4)
plt.show()

sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind="kde", levels=[.01, .05, .1, .8])
plt.show()

sns.displot(diamonds, x="price", y="clarity", log_scale=(True, False))
plt.show()

sns.displot(diamonds, x="color", y="clarity")
plt.show()

# Distribution visualization in other settings
sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")
plt.show()

sns.jointplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm", hue="species",
    kind="kde"
)
plt.show()

g = sns.JointGrid(data=penguins, x="bill_length_mm", y="bill_depth_mm")
g.plot_joint(sns.histplot)
g.plot_marginals(sns.boxplot)
plt.show()

sns.displot(
    penguins, x="bill_length_mm", y="bill_depth_mm",
    kind="kde", rug=True
)
plt.show()

sns.relplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")
sns.rugplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")
plt.show()

# Plotting many distributions
sns.pairplot(penguins)
plt.show()

g = sns.PairGrid(penguins)
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot, fill=True)
g.map_diag(sns.histplot, kde=True)
plt.show()
