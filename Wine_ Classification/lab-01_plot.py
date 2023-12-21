import numpy as np
from matplotlib import pyplot as plt

# 아래 코드를 실행하기 위해서는 pip로 seaborn과 pandas를 설치한다.

# pip install seaborn
# --> 위 커맨드로 pandas도 설치됨
import seaborn as sns
import pandas as pd

wine_data = pd.read_csv('C:/2023_python_ai/basicai_fa23/202021295_김희수_7주차과제/wine.csv', 'r', encodings = 'UTF-8')
sns.set(style="ticks", color_codes=True)
#iris = sns.load_dataset("iris")
g = sns.pairplot(wine_data, hue="wine", palette="husl")
plt.show()
wine_data.info()

'''
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

# Load data from CSV file
wine_data = pd.read_csv("wine_data.csv")

sns.set(style="ticks", color_codes=True)
g = sns.pairplot(wine_data, hue="Wine", palette="husl")
plt.show()
wine_data.info()
'''
