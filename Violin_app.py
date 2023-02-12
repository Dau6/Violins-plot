import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_y=[8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82, 5.68]
data_x=[10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0,4.0,12.0, 7.0,5.0]


fig, ax=plt.subplots()
ax.scatter(x=data_x, y=data_y, c="#E69F00")


#ax.set_xlabel("We should label the x-axis")
#ax.set_ylabel("We should label the y axis")
#ax.set_title("Some title")


plt.xlabel("We should label the x-axis")
plt.ylabel("We should label the y axis")
plt.title("Some title")

#plt.savefig("my_first_plot.png")




 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('C://Users//DELL//Downloads//gapminder_with_codes.csv')

data_2007 = data[data["year"]==2007]

fig, ax=plt.subplots()
ax.scatter(x=data_2007['gdpPercap'], y=data_2007['lifeExp'], alpha=0.5)
ax.set_xlabel('GDP (USD) per capita')
ax.set_ylabel('life expectancy (years)')




fig, ax=plt.subplots()

ax.scatter(x=data_2007['gdpPercap'], y=data_2007['lifeExp'], alpha=0.9)
ax.set_xscale("log")
ax.set_xlabel('GDP (USD) Per Capita')
ax.set_ylabel('Life expectancy (years)')




fig, ax=plt.subplots()
ax.scatter(x=data_2007['gdpPercap'], y=data_2007['lifeExp'], alpha=0.8)
ax.set_ylabel('Life expectancy (years)', fontsize=15)
ax.set_xlabel('GDP(USD) per Capita', fontsize = 15)

ax.tick_params(which='major', length=10)
#ax.tick_params(which ='minor', length=10)
#ax.tick_params(labelsize=15)





import numpy as np
import seaborn as sns

sns.set_theme()

# Create a random dataset across several variables
rs = np.random.default_rng(0)
n, p = 40, 8
d = rs.normal(0, 2, (n, p))
d += np.log(np.arange(1, p + 1)) * -5 + 10

# Show each distribution with both violins and points
sns.violinplot(data=d, palette="light:g", inner="points", orient="h")






import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#df = sns.load_dataset('data_2007')


data = pd.read_csv('C://Users//DELL//Downloads//gapminder_with_codes.csv')

data_2007 = data[data["year"]==2007]

#data_2007

#fig, ax = plt.subplots()
sns.violinplot(x=data_2007['gdpPercap'])
#, y=data_2007['lifeExp'])
sns.violinplot(y=data_2007['lifeExp'])


#sns.violinplot(data=df, x=data_2007['gdpPercap'], y=data_2007['lifeExp'])
plt.show()









