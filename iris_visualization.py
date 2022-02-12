import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


iris = pd.read_csv("./Iris.csv")
print(iris["Species"].value_counts())

iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
#plt.show()

#shows distribution with histograms
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)
#plt.show()


sns.FacetGrid(data=iris, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
#plt.show()

sns.boxplot(x="Species", y="SepalLengthCm", data=iris)
plt.show()

ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")

plt.show()


#violin_plot is combined sns.boxplot and sns.stripplot: denser regions of data are fatter and sparser are thinner
sns.violinplot(x="Species", y="SepalLengthCm", data=iris, size=6)
plt.show()


#shows density of species on ceratin feature
sns.FacetGrid(data=iris, hue="Species", size=5).map(sns.kdeplot, "PetalLengthCm").add_legend()
plt.show()


#now some plots for bivariate relations

sns.pairplot(data=iris.drop("Id", axis=1), hue="Species", size=4)
plt.show()


iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))

plt.show()


