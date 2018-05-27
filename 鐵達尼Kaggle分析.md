
# 鐵達尼Kaggle分析

<font size=2>&emsp;&emsp;這是我的第一個Kaggle專案，Titanic生存預測競賽是在Kaggle中是一個很經典的案例。藉由此練習與深入探討，我熟悉資料分析的流程，也對於python在機器學習的運用更加瞭解。經過一段時間的研究與努力，最終取得了Top 5%的成績，期待與大家討論交流。

## 分析大綱
> 1. 資料簡介
> 2. 數據清洗
> 3. 特徵工程
> 4. 建立模型
> 5. 繳交答案

## 1. 資料簡介

### 1.1 讀取資料與導入套件


```python
#導入所需套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#設定介面與字型
plt.style.use('ggplot')
# plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.family']='DFKai-SB' 
plt.rcParams['axes.unicode_minus']=False 

#設定展示欄位最大值
pd.set_option("display.max_columns",50) 

#載入訓練資料集
train = pd.read_csv('train.csv', encoding='big5')
#載入測試資料集
test = pd.read_csv('test.csv', encoding='big5')

#列出訓練資料集前5筆
print(train.head(5))

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }
    .dataframe thead th {
        text-align: left;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
#描述訓練資料集的資料特性
print(train.describe())
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }
    .dataframe thead th {
        text-align: left;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



### 1.2 資料的內容與類型

<font size=2> 此數據訓練集共有891位乘客，資料內容包含:
* 乘客編號　PassengerId: int 
* 存活與否　Survived: int
* 艙等　　　Pclass: int
* 姓名　　　Name: string
* 性別　　　Sex: string
* 年齡　　　Age: float
* 親戚人數　SibSp: int
* 至親人數　Parch: int
* 門票編號　Ticket: string
* 門票費用　Fare: float
* 船艙座位　Cabin: string
* 搭乘港口　Embarked: string


```python
#列出測試資料集前5筆
print(test.head(5))
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }
    .dataframe thead th {
        text-align: left;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



### 1.3 分析目標

<font size=2>&emsp;&emsp;從test資料集，我們可以看出相較於train資料集少了Survived的特徵屬性。此專案的分析目標是藉由搭乘鐵達尼的乘客所擁有的屬性去預測這名乘客最後是否存活，1代表存活(survived)，0代表死亡(did not survived)。所有的屬性都有可能是潛在的影響因子，最終須要使用test資料集去做預測，並提交至Kaggle官網確認預測準確率。

### 1.4 概覽資料


```python
f,ax = plt.subplots(3,4,figsize=(20,16))
sns.countplot('Pclass',data=train,ax=ax[0,0])
sns.countplot('Sex',data=train,ax=ax[0,1])
sns.boxplot(x='Pclass',y='Age',data=train,ax=ax[0,2])
sns.countplot('SibSp',hue='Survived',data=train,ax=ax[0,3],palette='husl')
sns.distplot(train['Fare'].dropna(),ax=ax[2,0],kde=False,color='b')
sns.countplot('Embarked',data=train,ax=ax[2,2])

sns.countplot('Pclass',hue='Survived',data=train,ax=ax[1,0],palette='husl')
sns.countplot('Sex',hue='Survived',data=train,ax=ax[1,1],palette='husl')
sns.distplot(train[train['Survived']==0]['Age'].dropna(),ax=ax[1,2],kde=False,color='r',bins=5)
sns.distplot(train[train['Survived']==1]['Age'].dropna(),ax=ax[1,2],kde=False,color='g',bins=5)
sns.countplot('Parch',hue='Survived',data=train,ax=ax[1,3],palette='husl')
sns.swarmplot(x='Pclass',y='Fare',hue='Survived',data=train,palette='husl',ax=ax[2,1])
sns.countplot('Embarked',hue='Survived',data=train,ax=ax[2,3],palette='husl')

ax[0,0].set_title('Total Passengers by Class')
ax[0,1].set_title('Total Passengers by Gender')
ax[0,2].set_title('Age Box Plot By Class')
ax[0,3].set_title('Survival Rate by SibSp')
ax[1,0].set_title('Survival Rate by Class')
ax[1,1].set_title('Survival Rate by Gender')
ax[1,2].set_title('Survival Rate by Age')
ax[1,3].set_title('Survival Rate by Parch')
ax[2,0].set_title('Fare Distribution')
ax[2,1].set_title('Survival Rate by Fare and Pclass')
ax[2,2].set_title('Total Passengers by Embarked')
ax[2,3].set_title('Survival Rate by Embarked')
```






![Imgur](https://i.imgur.com/Jauy60v.png)

## 2.數據清洗

<font size=2> &emsp;&emsp;在對於數據的內容與輪廓有些瞭解之後，我將對此數據進行一些篩選及填補，以及再做進一步的觀察，選出重要且具有代表性的特徵。</font>

### 2.1 離群值檢測

<font size=2>&emsp;&emsp;由於離群值往往會讓數據失真，因此在此使用離群值檢測，引用盒鬚圖的概念，把超過1.5倍IQR的數據篩選掉。


```python
from collections import Counter

def detect_outliers(df,n,features):
    outlier_indices = []
     # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.nanpercentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.nanpercentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1        
        # outlier step
        outlier_step = 1.5 * IQR        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])

train.loc[Outliers_to_drop] # Show the outliers rows
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }
    .dataframe thead th {
        text-align: left;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>745</th>
      <td>746</td>
      <td>0</td>
      <td>1</td>
      <td>Crosby, Capt. Edward Gifford</td>
      <td>male</td>
      <td>70.0</td>
      <td>1</td>
      <td>1</td>
      <td>WE/P 5735</td>
      <td>71.00</td>
      <td>B22</td>
      <td>S</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Charles Alexander</td>
      <td>male</td>
      <td>19.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.00</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>88</th>
      <td>89</td>
      <td>1</td>
      <td>1</td>
      <td>Fortune, Miss. Mabel Helen</td>
      <td>female</td>
      <td>23.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.00</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>159</th>
      <td>160</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Master. Thomas Henry</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.55</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>180</th>
      <td>181</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Miss. Constance Gladys</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.55</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>201</th>
      <td>202</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Mr. Frederick</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.55</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>324</th>
      <td>325</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Mr. George John Jr</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.55</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>341</th>
      <td>342</td>
      <td>1</td>
      <td>1</td>
      <td>Fortune, Miss. Alice Elizabeth</td>
      <td>female</td>
      <td>24.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.00</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>792</th>
      <td>793</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Miss. Stella Anna</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.55</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>846</th>
      <td>847</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Mr. Douglas Bullen</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.55</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>863</th>
      <td>864</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Miss. Dorothy Edith "Dolly"</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.55</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



<font size=2> **離群值觀察結果**

<font size=2>
* 共有10位乘客被判定為離群值
* 其中乘客編號28、89、342的票價太高
* 其餘的乘客都有8位親戚，而且名字裡都有Sage，明顯是個大家族


```python
# 將這10個離群數據剃除
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

#將訓練資料集與測試資料集合併(方便填補遺漏值)
full_data =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
```

### 2.2 填補遺漏值

<font size=2>&emsp;&emsp;一般數據常常會有遺漏值或是缺失值，這些資料的缺漏常常會造成數據的失真，甚至因為遺漏值太多，讓某些屬性的參考性大大降低，因此如何填補缺失值也是一門研究課題。由於填補遺漏值的動作，在所有資料都要執行，因此我們將train與test合併成full_data，方便填補遺漏值。


```python
#觀察是否為空值
full_data.isnull().sum()
```




    Age             256
    Cabin          1007
    Embarked          2
    Fare              1
    Name              0
    Parch             0
    PassengerId       0
    Pclass            0
    Sex               0
    SibSp             0
    Survived        418
    Ticket            0
    dtype: int64



<font size=2>&emsp;&emsp;由上表可以看出，綜合資料集的Age遺漏值有256個，Cabin的缺失值有1007個，Embarked缺失值有2個。由於Cabin遺漏比例過多，參考性過高(1007/1299)，預測時將不參考此特徵。以下將針對Age、Fare與Embarked，分別進行遺漏值填補。

### 2.2.1 Age

<font size=2>&emsp;&emsp;在處理遺漏值時，大多數的人都會「直接移除資料」或是用「平均值/眾數來填補遺漏值」，但這樣的做法並不推薦，前者會讓資料減少，後者可能會讓母體分布失真。因此，為了確保讓整體的Age分布能與原來的數據吻合，在這邊以"平均值+-標準差"之間的隨機數值填補Age遺漏值。(P.S.後續還有綜合其他概念的填補年齡方法，屆時再討論)


```python
#定義繪製新Age與原本Age資料分布圖
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values(填補前)')
axis2.set_title('New Age values(填補後)')

#分別計算平均值與標準差
average_age   = full_data["Age"].mean()
std_age       = full_data["Age"].std()
count_nan_age = full_data["Age"].isnull().sum()

#以隨機的方式填入
rand_1 = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)

full_data['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
full_data.loc[np.isnan(full_data["Age"]),"Age"] = rand_1
full_data['Age'] = full_data['Age'].astype(int)
full_data['Age'].hist(bins=70, ax=axis2)

plt.show()
```



![Imgur](https://i.imgur.com/Lbf0DBo.png)

<font size=2>&emsp;&emsp;由上圖我們可以看出，填補前與填補後的資料分布，大致上相同，沒有明顯分布變化。


```python
#繪製年齡分布與存活比例的關係圖
facet = sns.FacetGrid(full_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, full_data['Age'].max()))
facet.add_legend()

#繪製年齡與存活率分布圖
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = full_data[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)
plt.show()
```


![Imgur](https://i.imgur.com/SQKSFG7.png)



![Imgur](https://i.imgur.com/9mMwGtW.png)


<font size=2>&emsp;&emsp;我同時也利用「年齡與存活比例分佈圖」、「年齡與存活率分佈圖」觀察到以下現象:
> 1. 死亡比率最高的年齡區段為20~30歲    
> 2. 小孩與老人的存活率較高
<font size=2>&emsp;&emsp;合理推測，事發當時應該有「老人與小孩先走」的逃生順序，導致數據如此呈現。

### 2.2.2 Fare

<font size=2>&emsp;&emsp;Fare數據中，只有1個遺漏值，這個部份因為遺漏值很少，對於整體數據分佈影響不大，我們可以考慮使用簡單方式填補(平均數/眾數/中位數)。由下圖我們可以看得出Fare不是常態分佈，因此選擇中位數進行填補。


```python
#用中位數填補遺漏值
full_data["Fare"] = full_data["Fare"].fillna(full_data["Fare"].median())

# Fare分佈圖
g = plt.subplots(figsize=(8,5)) 
g = sns.distplot(full_data["Fare"], color="m", label="Skewness : %.2f"%(full_data["Fare"].skew()))
g = g.legend(loc='best',prop={'size':12})
plt.show()
```


![Imgur](https://i.imgur.com/CIoO3Km.png)


<font size=2>&emsp;&emsp;透過觀察，Fare分佈圖是有右偏斜Skewness情形發生，偏斜係數為4.51，右偏斜的特性為「平均數>中位數」。去偏斜是數據預處理流程中的一個操作步驟，應先將原分佈取自然對數後再rescale，如此所得到的是一個具有常態分佈特性的數據。數據成常態分佈後可以更便捷的進行統計推斷(t檢驗、ANOVA或者線性回歸分析)。


```python
# 對Skewness取Log,減少偏斜
full_data["Fare"] = full_data["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
g = plt.subplots(figsize=(8,5)) 
g = sns.distplot(full_data["Fare"], color="b", label="Skewness : %.2f"%(full_data["Fare"].skew()))
g = g.legend(loc="best",prop={'size':12})
plt.show()
```


![Imgur](https://i.imgur.com/Xz2VQpS.png)


<font size=2>&emsp;&emsp;經過去偏斜後，資料分佈比較像是常態分佈，偏斜係數也由4.51下降為0.58

### 2.2.3 Embarked

<font size=2>&emsp;&emsp;Embarked數據中，有2個遺漏值，觀察以下的資料集作圖，我們可以得知S港口是最多人上船的地方，但是存活率最低。以眾數的概念進行遺漏值的填補，對於整體的數據分布幾乎不產生影響。


```python
#Embarked數量與存活率分析
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

sns.countplot(x='Embarked', data=train, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=train, order=[1,0], ax=axis2)

embark_perc = train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)
plt.show()
```


![Imgur](https://i.imgur.com/vJIlUw6.png)



```python
#以眾數填補搭乘港口遺漏值
most_embarked = full_data.Embarked.value_counts().index[0]   
full_data.Embarked = full_data.Embarked.fillna(most_embarked)
```

## 3. 特徵工程

<font size=2>&emsp;&emsp;特徵工程是機器學習成功的關鍵，坊間常說「數據和特徵決定了機器學習的上限，而模型和算法只是逼近上限而已」。由此可見，特徵工程在機器學習中占有相當重要的地位。在實際應用當中，通常會把特徵工程看做是一個問題。事實上，特徵工程還包含3個子項目，以下將逐一探究。
> 1. 特徵建構: 從原始數據中人工的構建新的特徵，取代原始數據的特徵
> 2. 特徵提取: 將機器學習演算法不能識別的原始數據，轉化為演算法可以識別的特徵
> 3. 特徵選擇: 從所有的特徵中選擇一組最好的特徵集，捨去無關的特徵，保留相關性高的特徵

### 3.1 特徵建構

<font size=2>&emsp;&emsp;特徵建構主要是觀察原始數據，並將各個特徵做結合，建造超新的特徵。在產生新特徵的同時，也會捨棄原始的特徵，讓特徵維度降低，或維持在一定的水平，否則之後機器學習建模時很容易遇到「維度的詛咒」。

### 3.1.1 Family

<font size=2>&emsp;&emsp;我們目前有兩個相似的屬性Parch與Sibsp，分別是家庭人數與親戚人數，由於這兩個屬性是很類似的。我們可以將Parch與Sibsp相加，再加上自己1人，創建一個Family的新的屬性，當相關性高的屬性變少，有效降低維度，有利於未來的預測精準度。


```python
# 將Sib與Par相加，創建新的Family屬性
full_data['Family'] =  full_data["Parch"] + full_data["SibSp"] + 1

# 繪製出Family與Survived的關係分布圖
g = plt.subplots(figsize=(10,5)) 
family = full_data[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
g=sns.barplot(x='Family', y='Survived', data=family)
g = g.set_ylabel("Survival Probability")
plt.show()
```


![Imgur](https://i.imgur.com/zbOmt2y.png)


<font size=2>&emsp;&emsp;我們可以從此表看出，家族人數2-4人的生存率都大於50%，而單獨一人或家族人數大於5人的生存率都較低，推測是因為單獨1人不易得到他人協助；而家族人數過多，可能會有互相等待找人而錯失逃生的絕佳時機。因此我們可以將此特徵做分組，1人為"Single"，2-4人為"SmallF"，5人以上為"LargeF"。
<font size=2>&emsp;&emsp;這裡我引用的概念是「連續特徵的離散化」，一般機器學習很少直接將連續值作為邏輯回歸模型的特徵輸入，而是將連續特徵離散化為一系列0、1特徵或離散特徵，再導入模型。離散化的優點:
> 1. 離散特徵的增加和減少都很容易，易於模型的快速迭代
> 2. 離散化後的特徵對異常數據有很強的強健性
> 3. 有效簡化模型的作用，降低模型過擬合的風險


```python
#將連續型的Family人樹從連續轉成離散的三組
full_data['Single'] = full_data['Family'].map(lambda s: 1 if s == 1 else 0)
full_data['SmallF'] = full_data['Family'].map(lambda s: 1 if 2 <= s <= 4 else 0)
full_data['LargeF'] = full_data['Family'].map(lambda s: 1 if s >= 5 else 0)
```

### 3.1.2 Title

<font size=2>&emsp;&emsp;我這邊想探討姓名的稱謂，因為不同的稱謂可能代表不同的身分，假設他是船員，由於工作責任，可能逃生的優先順序就會比較後面，希望能藉此發現一些新的影響因子。


```python
# 分類稱謂
full_data_title = [i.split(",")[1].split(".")[0].strip() for i in full_data["Name"]]
full_data["Title"] = pd.Series(full_data_title)

#繪製稱謂分布圖
g = plt.subplots(figsize=(12,4)) 
g = sns.countplot(x="Title",data=full_data)
g = plt.setp(g.get_xticklabels(), rotation=45) 
plt.show()
```


![Imgur](https://i.imgur.com/xUn16q0.png)


<font size=2>&emsp;&emsp;這邊我想簡化Title的類別數量，因此我以將貴族相關稱謂的標記為"Royal"，船組人員或是稀有稱謂為"Rare"，年輕女性為"Miss"，已婚女性標記為"Mrs"，男性標記為"Mr"，小孩子標記為"Master"


```python
# 將稱謂分組，並給予值 
full_data["Title"] = full_data["Title"].replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
full_data["Title"] = full_data["Title"].replace(['Lady', 'the Countess','Countess','Sir'], 'Royal')
full_data["Title"] = full_data["Title"].replace(["Miss", "Mlle", "Ms"],'Miss')
full_data["Title"] = full_data["Title"].replace(["Mme", "Mrs"],'Mrs')
```


```python
#繪製稱謂數量統計圖

plt.figure(figsize=(8,4))
g1 = sns.countplot(full_data["Title"])
plt.show()
```


![Imgur](https://i.imgur.com/ovkIaQm.png)



```python
#繪製稱謂生存比例圖
g2 = sns.factorplot(x="Title",y="Survived",data=full_data,kind="bar",size=4, aspect=2.5)
g2 = g2.set_ylabels("survival probability")
plt.show()
```


![Imgur](https://i.imgur.com/SpehmbM.png)


<font size=2>&emsp;&emsp;藉由稱謂統計與存活分布圖，我們可以發現，貴族的生存率是100%，而女性與小孩都超過50%，男性與Rare的存活率都低於30%，這是我們之後分析的重要參考。

### 3.1.2.1 Age遺漏值填補進階

<font size=2>&emsp;&emsp;此處我們可以發掘Title與存活率是有相關性的，而且可以把資料分得更細，所以回到2.2.1節的Age遺漏值填補，可以依照不同的Title的類別，結合正負標準差的填補法，填補更準確的年齡值，增加Age屬性的可靠度。(P.S. 若執行此節方法，則2.2.1方法可跳過)


```python
# 以Title結合常態分布填補Age值
Tit=[0,1,2,3,4,5]
for i in Tit:
    average_age   = full_data.loc[full_data.Title==i,'Age'].mean()
    std_age       = full_data.loc[full_data.Title==i,'Age'].std()
    count_nan_age = full_data.loc[full_data.Title==i,'Age'].isnull().sum()

    rand_1 = np.random.randint(average_age - std_age, average_age + std_age)
    full_data.loc[(full_data["Age"].isnull())&(full_data["Title"]==i),'Age'] = rand_1

full_data['Age'] = full_data['Age'].astype(int) 
```

### 3.1.3 Person

<font size=2>&emsp;&emsp;我們在2.2.1節時，發現可能當時有「老人與小孩先走」的救援順序，因此我想將建立一個新的Person屬性，依性別填入"Male"、"Female"，低於6歲的人為"child"，高於60歲的人為"older"


```python
#將人分成male,female,child
full_data['Person'] =  full_data["Sex"]
full_data.loc[full_data['Age'] <= 6,'Person'] = 'child'
full_data.loc[full_data['Age'] >= 60,'Person'] = 'older'
```

### 3.1.4 TPP指標(Title+Pclass+Person)

<font size=2>&emsp;&emsp;到目前為止，我們所觀察的都是單一屬性與存活率的關係，這個前提是"每個屬性都是獨立的"，然而生活中往往很多屬性都有相互的關聯，因此我這邊採用交叉比對分類的方式，同時參考Title、Pclass、Person屬性，製作新的TPP指標，希望能有更好的預測效果。


```python
#TPP指標計算
TPP=full_data[full_data.Survived.notnull()].pivot_table(
    index=['Title','Pclass','Person'],values=['Survived']).sort_values('Survived',ascending=False)

#繪出TPP指標長條圖
TPP.plot(kind='bar',figsize=(16,6),color='Orange')
plt.xticks(rotation=40)
plt.axhline(0.8,color='r')
plt.axhline(0.5,color='r')
plt.annotate('80% survival rate',xy=(30,0.81),xytext=(32,0.85),arrowprops=dict(facecolor='#BA55D3',shrink=0.05))
plt.annotate('50% survival rate',xy=(32,0.51),xytext=(34,0.54),arrowprops=dict(facecolor='#BA55D3',shrink=0.05))
plt.show()
```


![Imgur](https://i.imgur.com/uJXkQnK.png)


<font size=2>&emsp;&emsp;利用TPP指標，我們可以很清楚地看到，在交叉比對下的存活率，我們別在80%與50%，劃出標準線。並同時進行分組，大於80%為1，50-80%之間的為2，0-50%之間的為3，其餘的為4。有了TPP指標，之後在導入模型時，就不必再同時參考Title、Pclass、Person，將三個屬性組合成一個新的屬性，這也是一種降低維度的方法。


```python
#將TPP屬性的分組並賦值
Persons=['child','male','female','older']
Titles = ['Mr','Mrs','Miss','Master','Rare','Royal']
for i in Persons:
    for j in range(1,4):
        for g in Titles:
            if full_data.loc[(full_data.Title==g)&(full_data.Pclass==j)&(full_data.Person==i)&(full_data.Survived.notnull()),'Survived'].mean()>=0.8:
                full_data.loc[(full_data.Title==g)&(full_data.Pclass==j)&(full_data.Person==i),'TPP']=1
            elif full_data.loc[(full_data.Title==g)&(full_data.Pclass==j)&(full_data.Person==i)&(full_data.Survived.notnull()),'Survived'].mean()>=0.5:
                full_data.loc[(full_data.Title==g)&(full_data.Pclass==j)&(full_data.Person==i),'TPP']=2
            elif full_data.loc[(full_data.Title==g)&(full_data.Pclass==j)&(full_data.Person==i)&(full_data.Survived.notnull()),'Survived'].mean()>=0:
                full_data.loc[(full_data.Title==g)&(full_data.Pclass==j)&(full_data.Person==i),'TPP']=3
            else: 
                full_data.loc[(full_data.Title==g)&(full_data.Pclass==j)&(full_data.Person==i),'TPP']=4

```

### 3.2 特徵提取

<font size=2>&emsp;&emsp;在特徵建構完成之後，接著要做特徵提取，「特徵提取」的首要任務是將變數從文字轉換成數字/序數值，以利統計與導入演算法模型。以下分成兩種類型:
> 1. 定性數據: 名目型，類別之間沒有自然順序或是大小(ex.Title、Sex)
> 2. 定量數據: 數值型，通常為連續型，可區分大小(ex.Fare、Age)


```python
print(full_data.head(3))
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }
    .dataframe thead th {
        text-align: left;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Family</th>
      <th>Single</th>
      <th>SmallF</th>
      <th>LargeF</th>
      <th>Title</th>
      <th>Person</th>
      <th>TPP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22</td>
      <td>NaN</td>
      <td>S</td>
      <td>1.981001</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Mr</td>
      <td>male</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>C85</td>
      <td>C</td>
      <td>4.266662</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Mrs</td>
      <td>female</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26</td>
      <td>NaN</td>
      <td>S</td>
      <td>2.070022</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Miss</td>
      <td>female</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



<font size=2>&emsp;&emsp;我們可以藉由上表的觀察，目前所擁有的所有特徵，並判斷需要進行特徵提取的特徵。以下是需要進行特徵提取的部分，Fare、Age可以從連續型轉換成離散，Pclass、Person可以使用「one-hot encoding」轉換，而Title可以對應到數值。

### 3.2.1 Cut(Fare, Age)

<font size=2>&emsp;&emsp;將Fare分成五等分，將Age分成六等分，精簡模型的數值分析。


```python
#計算Fare分5等分
full_data.FareCut=pd.qcut(full_data.Fare,4)
#Fare分組
full_data.loc[full_data.Fare<=2.06,'FareCut']=1
full_data.loc[(full_data.Fare>2.06)&(full_data.Fare<=2.35),'FareCut']=2
full_data.loc[(full_data.Fare>2.35)&(full_data.Fare<=3.05),'FareCut']=3
full_data.loc[(full_data.Fare>3.05)&(full_data.Fare<=3.68),'FareCut']=4
full_data.loc[full_data.Fare>3.68,'FareCut']=5

#將Age分成六等分
full_data.AgeCut=pd.cut(full_data.Age,5)
full_data.loc[full_data.Age<=5,'AgeCut']=1                           # Baby
full_data.loc[(full_data.Age>5)&(full_data.Fare<=12),'AgeCut']=2     # Child
full_data.loc[(full_data.Age>12)&(full_data.Fare<=24),'AgeCut']=3    # Teenager
full_data.loc[(full_data.Age>24)&(full_data.Fare<=40),'AgeCut']=4    # young adult
full_data.loc[(full_data.Age>40)&(full_data.Fare<=60),'AgeCut']=5    # Adult
full_data.loc[full_data.Age>60,'AgeCut']=6                           # Senior
```

### 3.2.2 Pclass & Person & Title

<font size=2>&emsp;&emsp;在Pclass中，我們可以看到主要分成1~3個等級，以1、2、3作為分類，但是這三個等級之間沒有數值關係，演算法會將Pclass3視為比Pclass2大，為了避免誤導演算法的執行，這邊常使用的解決方法為「one-hot encoding」技術，將每個名目類別都建立一個新的特徵，以下我們將Pclass分成三個新的類別Pclass_1、Pclass_2、Pclass_3。<br>
<font size=2>&emsp;&emsp;同時，我們可以觀察出Pclass_1的乘客的獲救率較高，右圖再加上性別的特徵綜合來看，Pclass1與2的女性乘客存活率高達90%以上，這可以做為之後特徵選擇的參考。


```python
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))
sns.barplot(x="Pclass", y="Survived", data=train,ax=axis1)
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train,ax=axis2);
plt.show()
```


![Imgur](https://i.imgur.com/Q32WX2p.png)



```python
#對Pclass使用one-hot encoding
full_data = pd.get_dummies(full_data, columns = ["Pclass"])
```


<font size=2>&emsp;&emsp;Person與Embarked可以沿用Pclass的想法，也使用「one-hot encoding」技術將它轉換成獨立的特徵。記得使用完one-hot encoding後，原始的特徵可以刪除，否則相關性太高的特徵會影響到模型預測。


```python
#對Person使用one-hot encoding
full_data = pd.get_dummies(full_data['Person'])

#對Embarked使用one-hot encoding
full_data = pd.get_dummies(full_data, columns = ["Embarked"], prefix="Em")
```

<font size=2>&emsp;&emsp;關於Title，因為類別有6個，若使用one-hot encoding技術，可能會創造出太多的特徵，因此此處使用一般mapping技巧，並把數量最多的Mr放於0，減少數值之間的影響。


```python
full_data["Title"] = full_data["Title"].map({"Mr":0, "Rare":1, "Master":2, "Miss":3, "Mrs":4, "Royal":5})
```

### 3.2.3 特徵選擇


```python
full_data.head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }
    .dataframe thead th {
        text-align: left;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Family</th>
      <th>Single</th>
      <th>SmallF</th>
      <th>LargeF</th>
      <th>Title</th>
      <th>TPP</th>
      <th>FareCut</th>
      <th>AgeCut</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>child</th>
      <th>female</th>
      <th>male</th>
      <th>older</th>
      <th>Em_C</th>
      <th>Em_Q</th>
      <th>Em_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22</td>
      <td>NaN</td>
      <td>1.981001</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>C85</td>
      <td>4.266662</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>2</td>
      <td>female</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26</td>
      <td>NaN</td>
      <td>2.070022</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<font size=2>&emsp;&emsp;「特徵選擇」的主要目標是找出與存活率相關性高的特徵，剔除無法萃取資訊或是重複性高的特徵，通常這個步驟不是一次到位，需要搭配相關性的探討。第一階段，先剔除不會使用到的特徵，以下是剃除的特徵：Cabin遺漏值太多，Name已經轉換成Title，Fare與Age已經轉換成Cut了，Parch與SibSp已經轉換成Family，而Family也被分成單身、小家庭、大家庭了，Sex已經轉換成Person了，Ticket資訊太亂所以捨棄，PassengerId沒有幫助。


```python
full_data.drop(labels = ["Cabin","Name","Age", "SibSp","Parch","Fare","PassengerId","Family","Ticket", "Sex"], axis = 1, inplace = True)
```


```python
full_data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }
    .dataframe thead th {
        text-align: left;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Family</th>
      <th>Single</th>
      <th>SmallF</th>
      <th>LargeF</th>
      <th>Title</th>
      <th>TPP</th>
      <th>FareCut</th>
      <th>AgeCut</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>child</th>
      <th>female</th>
      <th>male</th>
      <th>older</th>
      <th>Em_C</th>
      <th>Em_Q</th>
      <th>Em_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<font size=2>&emsp;&emsp;再次觀察目前留下來的特徵，可以看到上表的特徵型態都被轉換成簡單的數值，此時讓我們開始觀察這些特徵的相關性。我們可以使用相關性作圖(Pearson相關係數)，下圖中顏色較淺的部分為沒有關係的，顏色偏藍就是正相關，顏色偏紅就是負相關，而我們在意的是與Survived有相關性無論正負，若是值越接近0，即是要被剃除的特徵。


```python
colormap = plt.cm.RdBu
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=20)
sns.heatmap(full_data[['AgeCut','FareCut','Single','SmallF','LargeF','Title','TPP','Pclass_1','Pclass_2'
                       ,'Pclass_3','child','female','male','older','Em_C','Em_Q','Em_S','Survived']]
            .astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, 
            cmap=colormap, linecolor='white', annot=True)
plt.show()
```


![Imgur](https://i.imgur.com/gV1DrII.png)


<font size=2>&emsp;&emsp;與Survived最不相關的可能是Embarked(C、Q、S)，older、LargeF、child、Pclass的相關性也於0.15，而female與male與TPP的相關性高達90%，因此也建議刪除，只要留下TPP即可，否則可能會有overfitting的情形，此即為第二階段特徵篩選：


```python
full_data.drop(labels = ["Em_C","Em_Q","Em_S","older","female","male","child","Pclass_2","LargeF"], axis = 1, inplace = True)
```


```python
full_data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }
    .dataframe thead th {
        text-align: left;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Family</th>
      <th>Single</th>
      <th>SmallF</th>
      <th>Title</th>
      <th>TPP</th>
      <th>FareCut</th>
      <th>AgeCut</th>
      <th>Pclass_1</th>
      <th>Pclass_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 4. 建立模型

<font size=2>&emsp;&emsp;經過前面的清洗與特徵工程的調整後，現在這個階段，我們要將full_data再次分成test與train，並進行Cross-validation交叉驗證，並套用scikit-learn演算法。

### 4.1 檔案分割

<font size=2>&emsp;&emsp;為了之後訓練train資料集，需要先得到train與test資料集，並將test資料集分成測試答案y與測試題目X，去驗證自身準確度。


```python
# 得到train與test資料集
train_len = len(train)
train = full_data[:train_len]
test = full_data[train_len:]
test = test.drop(labels =['Survived'],axis = 1)

# 將train資料集與Survived答案分開，y是測試答案，X是測試題目
train["Survived"].astype(int)
y = train["Survived"]
X = train.drop(labels =["Survived"], axis = 1)
```

### 4.2 Cross-validation交叉驗證

<font size=2>&emsp;&emsp;交叉驗證(Cross-validation)主要用於模型訓練或建模應用中，如分類預測、PCR、PLS回歸建模等。在給定的樣本空間中，拿出大部分樣本作為訓練集來訓練模型，剩餘的小部分樣本使用剛建立的模型進行預測，並求這小部分樣本的預測誤差或者預測精度。使用交叉驗證方法的目的主要有3個：
> 1. 從有限的學習數據中獲取儘可能多的有效信息
> 2. 交叉驗證從多個方向開始學習樣本的，可以有效的避免陷入局部最小值
> 3. 可以在一定程度上避免過擬合問題


```python
#使用CV方法，並設test_size為0.3(70%測試，30%驗證)
from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.3)
```

<font size=2>&emsp;&emsp;此處導入「正規化」概念，在訓練模型的時候，要輸入特徵。對於同一個特徵，不同的樣本中的取值可能會相差非常大，一些異常小或異常大的數據會誤導模型的正確訓練；另外，如果數據的分佈很分散也會影響訓練結果。以上兩種方式都體現在方差會非常大。此時，我們可以將特徵中的值進行標準差正規化，即轉換為均值為0，方差為1的常態分佈。


```python
#進行正規化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_normalized = sc.transform(X_train)
X_cv_normalized = sc.transform(X_cv)
```

### 4.3 引入作圖程式

<font size=2>&emsp;&emsp;以下引入scikit-learn官網所提供的作圖程式，讓我們站在巨人們的肩膀上看得更遠吧！


```python
# Scikit-Learn 官網作圖函式
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
   
    plt.figure(figsize=(10,6))  #調整作圖大小
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
```

    Automatically created module for IPython interactive environment
    


```python
#導入模糊矩陣作圖程式
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```

### 4.4 代入演算法

<font size=2>&emsp;&emsp;這邊我代入許多線上常用的演算法，去觀察哪個演算法在此題的表現較好，將進行CV後的結果視覺化。


```python
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from xgboost import XGBClassifier


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)

# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(XGBClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train_normalized, y_train.values.ravel(), scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","XGBC","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

plt.figure(figsize=(12,6))
g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
plt.show()
```


![Imgur](https://i.imgur.com/H6dYcOf.png)


<font size=2>&emsp;&emsp;由上圖我們可以看出其中SVC、AdaBoost演算法的準確率最高，而XGBC演算法最穩定。在代入模型時需要小心訓練資料集模型的準確度過高，反而可能會導致模型太過於偏向訓練資料集(train)，也就是所謂的overfitting過度擬合，一個荒謬的模型只要足夠複雜，是可以完美地適應資料。有句機器學習的反諷名言，「如果你花足夠的時間去拷問資料,它就會如你所願招供」。但是這樣做出來的模型是沒有意義的，而且真實應用上的準確度會降低許多。現在就讓我們來看看它們有沒有overfitting的情形吧~

<font size=2>&emsp;&emsp;這裡採用Grid search進行最優參數選擇，Grid search能夠自動選擇最優參數。參數優化的數量越多，則所需時間越長，因此需要對演算法與各個參數進行深入探討，明確參數調整方向，可以節省許多時間。


```python
## SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [10**i for i in range(-2, 2)],
                  'C': [10**i for i in range(-1, 2)]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 2, verbose = 1)
gsSVMC.fit(X_train_normalized, y_train.values.ravel())
SVMC_best = gsSVMC.best_estimator_

# Best score
print(gsSVMC.best_score_)
print(gsSVMC.best_params_)
```

    Fitting 10 folds for each of 49 candidates, totalling 490 fits
    

    [Parallel(n_jobs=2)]: Done 180 tasks      | elapsed:    6.0s
    

    0.837662337662
    {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
    

    [Parallel(n_jobs=2)]: Done 490 out of 490 | elapsed:   16.4s finished
    


```python
#XGB classifier

XGBC = XGBClassifier()
XGBC_param_grid = {
        'learning_rate': [0.05],
        'max_depth': list(range(4, 10)),
        'n_estimators': [10, 15, 25, 50, 100, 250],
        'gamma': [ 0.01, 0.1, 1, 10]
    }
gsXGBC = GridSearchCV(XGBC, param_grid = XGBC_param_grid, cv=kfold, scoring="accuracy", n_jobs=2, verbose = 1)

gsXGBC.fit(X_train_normalized, y_train.values.ravel())

XGBC_best = gsXGBC.best_estimator_

# Best score
print(gsXGBC.best_score_)
print(gsXGBC.best_params_)
```

    Fitting 10 folds for each of 1152 candidates, totalling 11520 fits
    

    [Parallel(n_jobs=2)]: Done 331 tasks      | elapsed:   14.7s
    [Parallel(n_jobs=2)]: Done 1139 tasks      | elapsed:   53.6s
    [Parallel(n_jobs=2)]: Done 2139 tasks      | elapsed:  1.8min
    [Parallel(n_jobs=2)]: Done 3539 tasks      | elapsed:  2.9min
    [Parallel(n_jobs=2)]: Done 5339 tasks      | elapsed:  4.4min
    [Parallel(n_jobs=2)]: Done 7539 tasks      | elapsed:  6.3min
    [Parallel(n_jobs=2)]: Done 10139 tasks      | elapsed:  8.5min
    [Parallel(n_jobs=2)]: Done 11520 out of 11520 | elapsed:  9.7min finished
    

    0.837662337662
    {'colsample_bylevel': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.01, 'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 250}
    


```python
# Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[10, 25, 50, 100],
              "learning_rate":  [0.001, 0.05, 0.1, 0.5, 1, 2]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 2, verbose = 1)

gsadaDTC.fit(X_train_normalized, y_train.values.ravel())

ada_best = gsadaDTC.best_estimator_

print(gsadaDTC.best_score_)
print(gsadaDTC.best_params_)
```

    Fitting 10 folds for each of 240 candidates, totalling 2400 fits
    

    [Parallel(n_jobs=2)]: Done 131 tasks      | elapsed:    9.7s
    [Parallel(n_jobs=2)]: Done 731 tasks      | elapsed:   56.0s
    [Parallel(n_jobs=2)]: Done 1731 tasks      | elapsed:  2.2min
    

    0.827922077922
    {'algorithm': 'SAMME', 'base_estimator__criterion': 'entropy', 'base_estimator__splitter': 'random', 'learning_rate': 2, 'n_estimators': 25}
    

    [Parallel(n_jobs=2)]: Done 2400 out of 2400 | elapsed:  3.2min finished
    

<font size=2>&emsp;&emsp;在取得這三個演算法最佳化參數後，可以進行繪圖，確認個別的學習曲線走勢。


```python
g = plot_learning_curve(SVMC_best,"SVMC",sc.transform(X), y.values.ravel(),cv=kfold)
g = plot_learning_curve(XGBC_best,"XGBC",sc.transform(X), y.values.ravel(),cv=kfold)
g = plot_learning_curve(ada_best,"ada",sc.transform(X), y.values.ravel(),cv=kfold)
plt.show()
```


![Imgur](https://i.imgur.com/aXwOKUz.png)



![Imgur](https://i.imgur.com/m3e3P29.png)



![Imgur](https://i.imgur.com/O1F6Jy1.png)


### 4.5 綜合演算法votingC

<font size=2>&emsp;&emsp;在將這三個演算法，放入votingC的綜合演算法，也就是讓這三個演算法進行投票，決定最終的答案。讓我們先觀察votingC的學習曲線。


```python
votingC = VotingClassifier(estimators=[ ("SVMC", SVMC_best),
('XGBC', XGBC_best),('ada',ada_best)], voting='soft', n_jobs=2,weights=[1, 1, 1])

votingC = votingC.fit(X_train_normalized, y_train.values.ravel())

print(votingC.score(X_train_normalized, y_train.values.ravel()))
print(votingC.fit(X_train_normalized, y_train.values.ravel()))
```

    0.86525974026
    VotingClassifier(estimators=[('SVMC', SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
      max_iter=-1, probability=True, random_state=None, shrinking=True,
      tol=0.001, verbose=False)), ('XGBC', XGBClassifier(base_score=0.5, colsample_bylevel=...=None,
                splitter='random'),
              learning_rate=2, n_estimators=25, random_state=7))],
             flatten_transform=None, n_jobs=2, voting='soft',
             weights=[1, 1, 1])
    


```python
#繪製學習曲線
from plot_learning_curve import plot_learning_curve
from sklearn.model_selection import KFold

cv = KFold(n_splits=11, random_state=2, shuffle=True)   
estimator = votingC
sc.fit(X)
plot_learning_curve(estimator, "votingC", sc.transform(X), y.values.ravel(), cv=kfold, train_sizes=np.linspace(0.2, 1.0, 5))
plt.show()
```

    None
    


![Imgur](https://i.imgur.com/MOYMq8K.png)


<font size=2>&emsp;&emsp;要確認模型的預測的真實狀況，可以利用混淆矩陣去觀察。


```python
#繪製混淆矩陣
from sklearn import metrics
estimator = votingC
print(metrics.classification_report(y_cv, estimator.predict(X_cv_normalized)))
print(metrics.confusion_matrix(y_cv, estimator.predict(X_cv_normalized)))

from plot_confusion_matrix import plot_confusion_matrix
cnf_matrix = metrics.confusion_matrix(y_cv,estimator.predict(X_cv_normalized))
target_names = ['not survival', 'survival']
plot_confusion_matrix(cnf_matrix, classes=target_names) 
plt.show()
```

                 precision    recall  f1-score   support
    
            0.0       0.82      0.90      0.86       163
            1.0       0.80      0.69      0.74       101
    
    avg / total       0.82      0.82      0.82       264
    
    [[146  17]
     [ 31  70]]
    Confusion matrix, without normalization
    [[146  17]
     [ 31  70]]
    


![Imgur](https://i.imgur.com/f9I8ylI.png)


<font size=2>&emsp;&emsp;混淆矩陣主要判斷依據是f1 score越接近1越好。這次用train的預測成果，有31個存活者被我誤判成死亡，而有17個罹難者，我以為他們是倖存的，由此我可以做模型的調整。

## 5. 繳交答案

<font size=2>&emsp;&emsp;花了那麼多時間研究，終於要面對預測成果了，就讓我們輸出答案，上傳到Kaggle看看評分吧！


```python
test2 = pd.read_csv('test.csv', encoding='big5')
IDtest = test2["PassengerId"]
```


```python
votingC = votingC.fit(sc.transform(X), y.values.ravel())
X_test_normalized = sc.transform(test)
predictions = votingC.predict(X_test_normalized)

PassengerId =np.array(IDtest).astype(int)
my_solution = pd.DataFrame(predictions, PassengerId, columns = ["Survived"])
print(my_solution.head())
print(my_solution.shape)
my_solution.to_csv("my_solution_1.csv", index_label = ["PassengerId"])
```

         Survived
    892       0.0
    893       0.0
    894       0.0
    895       0.0
    896       1.0
    (418, 1)
    


![Imgur](https://i.imgur.com/yZd8hvs.png)

<font size=2>&emsp;&emsp;輸出結果的準確度為0.79425，大約為1700多名，因為是Demo版本，所以沒有突破我之前最高達到的0.82775的準確率，當然其中還有許多地方可以研究，Kaggle上也有許多大神可以學習，希望能彼此互相砥礪精進。



