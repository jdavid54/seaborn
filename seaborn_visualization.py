import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

# https://seaborn.pydata.org/tutorial/relational.html#emphasizing-continuity-with-line-plots

tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", data=tips);
plt.show()

sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips);
plt.show()

sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker", data=tips);
plt.show()

sns.relplot(x="total_bill", y="tip", hue="smoker", style="time", data=tips);
plt.show()

sns.relplot(x="total_bill", y="tip", hue="size", data=tips);
plt.show()

sns.relplot(x="total_bill", y="tip", hue="size", palette="ch:r=-.5,l=.75", data=tips);
plt.show()

sns.relplot(x="total_bill", y="tip", size="size", data=tips);
plt.show()

sns.relplot(x="total_bill", y="tip", size="size", sizes=(15, 200), data=tips);
plt.show()

# Emphasizing continuity with line plots
df = pd.DataFrame(dict(time=np.arange(500),
                       value=np.random.randn(500).cumsum()))
g = sns.relplot(x="time", y="value", kind="line", data=df)
g.fig.autofmt_xdate()
plt.show()

df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=["x", "y"])
sns.relplot(x="x", y="y", sort=False, kind="line", data=df);
plt.show()


# Aggregation and representing uncertainty
fmri = sns.load_dataset("fmri")
sns.relplot(x="timepoint", y="signal", kind="line", data=fmri);
plt.show()

sns.relplot(x="timepoint", y="signal", ci=None, kind="line", data=fmri);
plt.show()

sns.relplot(x="timepoint", y="signal", kind="line", ci="sd", data=fmri);
plt.show()

sns.relplot(x="timepoint", y="signal", estimator=None, kind="line", data=fmri);
plt.show()

# Plotting subsets of data with semantic mappings
sns.relplot(x="timepoint", y="signal", hue="event", kind="line", data=fmri);
plt.show()

sns.relplot(x="timepoint", y="signal", hue="region", style="event",
            kind="line", data=fmri);
plt.show()

sns.relplot(x="timepoint", y="signal", hue="region", style="event",
            dashes=False, markers=True, kind="line", data=fmri);
plt.show()


sns.relplot(x="timepoint", y="signal", hue="event", style="event",
            kind="line", data=fmri);
plt.show()

sns.relplot(x="timepoint", y="signal", hue="region",
            units="subject", estimator=None,
            kind="line", data=fmri.query("event == 'stim'"));
plt.show()

dots = sns.load_dataset("dots").query("align == 'dots'")
sns.relplot(x="time", y="firing_rate",
            hue="coherence", style="choice",
            kind="line", data=dots);
plt.show()

palette = sns.cubehelix_palette(light=.8, n_colors=6)
sns.relplot(x="time", y="firing_rate",
            hue="coherence", style="choice",
            palette=palette,
            kind="line", data=dots);
plt.show()

from matplotlib.colors import LogNorm
palette = sns.cubehelix_palette(light=.7, n_colors=6)
sns.relplot(x="time", y="firing_rate",
            hue="coherence", style="choice",
            hue_norm=LogNorm(),
            kind="line",
            data=dots.query("coherence > 0"));
plt.show()

sns.relplot(x="time", y="firing_rate",
            size="coherence", style="choice",
            kind="line", data=dots);
plt.show()

sns.relplot(x="time", y="firing_rate",
           hue="coherence", size="choice",
           palette=palette,
           kind="line", data=dots);
plt.show()

# Plotting with date data
df = pd.DataFrame(dict(time=pd.date_range("2017-1-1", periods=500),
                       value=np.random.randn(500).cumsum()))
g = sns.relplot(x="time", y="value", kind="line", data=df)
g.fig.autofmt_xdate()
plt.show()

sns.relplot(x="total_bill", y="tip", hue="smoker",
            col="time", data=tips);
plt.show()

sns.relplot(x="timepoint", y="signal", hue="subject",
            col="region", row="event", height=3,
            kind="line", estimator=None, data=fmri);
plt.show()

sns.relplot(x="timepoint", y="signal", hue="event", style="event",
            col="subject", col_wrap=5,
            height=3, aspect=.75, linewidth=2.5,
            kind="line", data=fmri.query("region == 'frontal'"));
plt.show()

