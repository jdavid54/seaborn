import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(color_codes=True)

# Visualizing regression models
tips = sns.load_dataset("tips")

sns.regplot(x="total_bill", y="tip", data=tips);
plt.show()

sns.lmplot(x="total_bill", y="tip", data=tips);
plt.show()

sns.lmplot(x="size", y="tip", data=tips);
plt.show()

sns.lmplot(x="size", y="tip", data=tips, x_jitter=.05);
plt.show()

sns.lmplot(x="size", y="tip", data=tips, x_estimator=np.mean);
plt.show()

# Fitting different kinds of models
anscombe = sns.load_dataset("anscombe")
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'I'"),
           ci=None, scatter_kws={"s": 80});
plt.show()

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
           ci=None, scatter_kws={"s": 80});
plt.show()


sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
           order=2, ci=None, scatter_kws={"s": 80});
plt.show()

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),
           ci=None, scatter_kws={"s": 80});
plt.show()

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),
           robust=True, ci=None, scatter_kws={"s": 80});
plt.show()

