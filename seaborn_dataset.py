import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

# https://seaborn.pydata.org/tutorial/data_structure.html

flights = sns.load_dataset("flights")
print(flights.head())

sns.relplot(data=flights, x="year", y="passengers", hue="month", kind="line")
plt.show()

flights_wide = flights.pivot(index="year", columns="month", values="passengers")
print(flights_wide.head())

sns.relplot(data=flights_wide, kind="line")
plt.show()

sns.relplot(data=flights, x="month", y="passengers", hue="year", kind="line")
plt.show()

sns.relplot(data=flights_wide.transpose(), kind="line")
plt.show()

sns.catplot(data=flights_wide, kind="box")
plt.show()

# messy data
anagrams = sns.load_dataset("anagrams")
print(anagrams)

anagrams_long = anagrams.melt(id_vars=["subidr", "attnr"], var_name="solutions", value_name="score")
print(anagrams_long.head())

sns.catplot(data=anagrams_long, x="solutions", y="score", hue="attnr", kind="point")
plt.show()

flights_dict = flights.to_dict()
sns.relplot(data=flights_dict, x="year", y="passengers", hue="month", kind="line")
plt.show()


flights_avg = flights.groupby("year").mean()
sns.relplot(data=flights_avg, x="year", y="passengers", kind="line")
plt.show()

year = flights_avg.index
passengers = flights_avg["passengers"]
sns.relplot(x=year, y=passengers, kind="line")
plt.show()

sns.relplot(x=year.to_numpy(), y=passengers.to_list(), kind="line")
plt.show()

flights_wide_list = [col for _, col in flights_wide.items()]
sns.relplot(data=flights_wide_list, kind="line")
plt.show()


two_series = [flights_wide.loc[:1955, "Jan"], flights_wide.loc[1952:, "Aug"]]
sns.relplot(data=two_series, kind="line")
plt.show()

two_arrays = [s.to_numpy() for s in two_series]
sns.relplot(data=two_arrays, kind="line")
plt.show()

two_arrays_dict = {s.name: s.to_numpy() for s in two_series}
sns.relplot(data=two_arrays_dict, kind="line")
plt.show()

flights_array = flights_wide.to_numpy()
sns.relplot(data=flights_array, kind="line")
plt.show()