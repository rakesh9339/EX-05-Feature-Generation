# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
```
Program Developed: Rakesh J.S
Register number:212222230115
```
```
Data.csv :
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```
# Encoding.csv :
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
```
# Titanic.csv :
```
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```
# OUPUT
## Data.csv :
## Initial Dataset:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/16290f4c-e02d-4047-bc72-a10754f38822)


##Binary Encoding:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/bc28a1a2-a501-49ba-a2a8-4bd72d5596d5)
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/6e366be0-d66e-467a-9e22-27f7d4f21b1d)




##Encoded Dataset:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/7e03518f-4ddd-4150-8044-e8c446a51b72)


##Data Scaling using MinMaxScaler:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/e30fbb91-3e47-493f-8488-c38e7f1883fb)


##Data Scaling using StandardScaler:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/f712fada-f225-41b0-8803-c1c4c271fea9)


##Data Scaling using MaxAbsScaler:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/70215714-bb1d-4751-b0f7-013f5ded5e5a)


##Data Scaling using RobustScaler:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/887ed8a2-7667-4117-8df5-9b486509b975)


##Encoding.csv :
##Initial Dataset:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/5ac7ec98-6510-4054-82e0-c02c0be42b2f)


##Binary Encoding:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/9d824f5d-e481-4a0c-ac64-c92b84e25b0f)
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/1af5d7ba-63bc-4551-b999-161dd58ac05c)




##Encoded Dataset:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/369686e8-cf6e-47aa-b096-d60369283a14)


##Data Scaling using MinMaxScaler:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/a8aeecf7-1f5b-4853-ac59-b33cdf496438)


##Data Scaling using StandardScaler:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/748e055c-4d96-4e39-8d04-112ce7e902db)


##Data Scaling using MaxAbsScaler:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/27714e81-b98d-4e2a-b257-505d29ae3557)


##Data Scaling using RobustScaler:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/02ce2f9e-593e-483e-b1a1-7a8b47d9445c)


##Titanic.csv :
##Initial Dataset:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/a46bb7eb-9296-47de-a196-58378f79aad1)

##Data cleaning before encoding:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/aeecaad9-8688-4ad7-a4a1-171ab4c1d4d7)






##Cleaned Dataset:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/d018b82e-3ab7-41ce-a58e-e6fe4cfa1d4c)


##Binary Encoding:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/24a4a096-53ad-4779-9f47-db564e32debe)


##Encoded Dataset:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/58c1b1de-8a90-4d5e-953e-4441ab5fcd9f)


##Data Scaling using MinMaxScaler:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/dc8e309f-8d34-4ae9-b01a-f82d2b13a359)


##Data Scaling using StandardScaler:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/6e9d06ea-c446-4a43-bf13-97a45110efb1)


##Data Scaling using MaxAbsScaler:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/1fe69427-deaf-4b56-a2da-d130b27c11dd)


##Data Scaling using RobustScaler:
![image](https://github.com/rakesh9339/EX-05-Feature-Generation/assets/121115650/92057105-f490-47d0-94e0-65119c95d351)


# RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.

