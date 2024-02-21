# Titanic-project

##  <font color=blue> Python Unit Summary </font>  

### Titanic database

### Introduction
### In this unit summary, we will analyze the Titanic data and see if there is a correlation between the chances of survival and certain characteristics of each passenger (gender, cabin class, ticket price, etc.)

### Part 1 - Guided Data Analysis

###  1. Import the required libraries to the notebook
(If you later need to use additional libraries, go back to this cell, add the
libraries here and run the cell again)

Explanation: For the sake of order and organization, it is recommended that all
libraries imported into the notebook be at the top of the notebook


import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

### 2. Import the titanic.csv data file into the notebook.
• PassengerId: Id of every passenger

• Pclass: There are 3 classes of passengers. Class1 (best one), Class2 and Class3

• Survived: This feature has a value 0 and 1. 0 = not survived, 1 for =survived

• Name: Name of passenger

• Sex: Gender of passenger

• Age: Age of passenger

• SibSp: number of siblings or spouses aboard the Titanic

• Parch: number of parents/children aboard the Titanic

• Ticket: Ticket number

• Fare: passenger fare

• Cabin: The cabin number of passenger

• Embarked: port of embarkation C=Cherbourg, Q=Queenstown, S=Southampton

• home.dest: The passenger destination - this column is missing!!

df=pd.read_csv(r'C:/Neytech Academy/Python/FUP/titanic_train.csv')

 - #### By typing the name that we have assigned to the Data Frame, we get the first 5 rows and the last 5 rows.

df

### 3. Using the methods we learned, get to know the df:
• Check for missing data in the columns

• If there is missing data, does it make sense for these columns to have
missing data? How much data is missing? And will the lack of data affect
the analysis?

 - #### head() method - we can see the first n records in the Data Frame, by default n equals 5 

df.head(3)

 - #### tail() method - we can see the last n records in the Data Frame, by default n equals 5

df.tail(3)

 - #### type() method - we can check the data type of a variable or value

type(df['PassengerId'])

 - #### shape() method - we have as output a tuple variable, with the dimension of the Data Frame (rows, columns)

df.shape

####  <font color=blue> <ins>Conclusion</ins>: The Titanic Data Frame that we are analysing has 891 rows and 12 columns</font>  

 - #### info() method gives information about index, column names, non-null values and memory information

df.info()

####  <font color=blue> <ins>Conclusion</ins>: The columns 'Age', 'Cabin' and 'Embarked' have null-values </font>  

 - #### isnull() is a method that checks for missing values

#  using the isnull() method
df.isnull().sum()

#### <font color=blue> <ins>Conclusion</ins>: The columns 'Age' has 177 missing values, column 'Cabin' has 687 missing values and column 'Embarked' has 2 missing values.</font> 

 - #### descriebe() method gives the statistical description of columns with number values 

df.describe()

#### <font color=blue> <ins>Conclusion</ins>: In the Titanic Data Frame we have data for 891 Passengers, from which only 38,38% Survived. 
#### <font color=blue> The youngest passenger had only 5 months old and the oldest passenger had 80 years old. 
#### <font color=blue> The Passenger average age was 29 years old. 
#### <font color=blue> We can say that obviously most of the passengers were young people. </font>

 - #### nunique() - method that gives us the number of unique values each column had, to understand the categorical and non-categorical features

# Finding the number of unique values in each of the columns 
df.nunique()

 #### <font color=blue> <ins>Conclusion</ins>: The Series ‘Sex’ and ‘Survived’ in Data Frame have two possible values, the Series ‘Embarked’ and ‘Pclass’ have three possible values.</font>

 - #### value_counts() - method that gives us the number of occurencies of each value for a Serie

# Checking how many male and female were on board
df['Sex'].value_counts()

 #### <font color=blue> <ins>Conclusion</ins>: Most of the passengers on board the Titanic were men.</font>

# Checking how many passengers in DF survived; 0 = not survived, 1 for = survived 
df['Survived'].value_counts()

#### <font color=blue> <ins>Conclusion</ins>: Most of the passengers on board the Titanic did not survived form the sinking of the ‘Unsinkable’ ship.</font>

# Checking the passenger distribution by class.
df['Pclass'].value_counts()

#### <font color=blue> <ins>Conclusion</ins>: The number of passenger in the third class was higher than the number of passengers in the first and second class combined.</font>

# SibSp: number of siblings or spouses aboard the Titanic
df['SibSp'].value_counts()

# Parch: number of parents/children aboard the Titanic
df['Parch'].value_counts()

#### <font color=blue> <ins>Conclusion</ins>: Most passengers were traveling alone.</font>

### 4. Delete the cabin column from the df.

 - #### drop(columns=[ ], index=[ ], inplace=True/False) - function applied to the Data Frame that delets rows or columns

df.drop(columns='Cabin',axis=1,inplace=True)

# Checking the Data Frame after deleting column 'Cabin'
df

#### <font color=blue> <ins>Conclusion</ins>: The 'Cabin' column was missing 77% of the values. The values could not be obtained from another source, and this column was deleted. </font>

### 5. Update the missing values in the age column, for women, use the median value of all female passengers.
 - #### The 'Age' column is missing 19,86% of the values. 
 - #### Missing values will be filled in with the average age value for each passenger category according to gender.

# Creating a mask for female passengers, filter the data according to the gender of the passenger.
mask_female = df['Sex'] == 'female'

# Checking that the mask works properly
df[mask_female]

#### <ins>Conclusion</ins>: The mask works properly, we have 314 records female passenger in our mask, the same number that we get when we apply the value_counts() method on the 'Sex' Serie.

# Counting the missing values in the 'Age' Serie for the female passengers.
df[mask_female]['Age'].isnull().sum()

# Finding median 'Age' value for the female category
f_median=df.loc[mask_female,'Age'].median()
f_median

# Replacing null values for Column 'Age'  with the median value
df.loc[mask_female,'Age']=df.loc[mask_female,'Age'].fillna(f_median)

# Checking whether there are any null value for female passenger in the 'Age' Serie.
df[mask_female].isna().sum()

#### <font color=blue> <ins>Conclusion</ins>: The 'Age' column for female passenger was missing 53 records. Missing values were filled in with the female 'Age' median value. </font>

### 6. Update the missing values in the age column for men, use the median value of all male passengers

# Creating a mask for male passengers, filter the data according to the gender of the passenger.
mask_male=df['Sex']=='male'

# Checking that the mask works properly
df[mask_male]

#### <ins>Conclusion</ins>: The mask works properly, we have 577 records for male passenger in our mask, the same number that we get when we apply the value_counts() method on the 'Sex' Serie.

# Counting the missing value for the mask_male
df[mask_male].isna().sum()

# Finding median 'Age' value for the male category
m_median=df.loc[mask_male,'Age'].median()
m_median

# Replacing null values for Column 'Age'  with the median value
df.loc[mask_male,'Age']=df.loc[mask_male,'Age'].fillna(m_median)

# Checking whether the replacment was done
df[mask_male].isna().sum()

#### <font color=blue> <ins>Conclusion</ins>: The 'Age' column for male passenger was missing 124 records. Missing values were filled in with the male 'Age' median value. </font>

### 7. Fill in the missing value in the embarked column with the most commonly appearing value (the most repeated value in this column)

# Finding the most repeated value in the column using value_counts() method on the 'Embarked' column
# The most repeated value is S=Southampton 
df['Embarked'].value_counts()

# Retaining the most commonly appearing value in a variable
embarked_common=df['Embarked'].value_counts().head(1).index[0]
embarked_common

# Finding the 2 missing values
mask_embarked=df['Embarked'].isna() == True
df.loc[mask_embarked]

# Replacing the 2 missing value in the 'Embarked' column with the most commonly appearing value
df['Embarked'].fillna(embarked_common,inplace=True)

# Checking if the replacement was done correctly
df.loc[mask_embarked]

#### <font color=blue> <ins>Conclusion</ins>: The 'Embarked' was missing 2 records. Missing values were filled in with the most commonly appearing value. </font>

### 8. Create a chart to show the distribution of passengers by gender. Choose a chart that suits you best, and design it so it is clear and presentable.

# Distribution of passengers by gender
df['Sex'].value_counts()

df.groupby('Sex')['PassengerId'].count().plot.pie(labels=('Female','Male'),
                                  autopct='%1.1f%%',
                                  pctdistance=0.65,
                                  shadow=True,
                                  explode=(0.1,0),
                                  figsize=(5,5),
                                  title='Passenger Distribution by Gender',
                                  ylabel='')

#### <font color=blue> <ins>Conclusion</ins>: Most of the passengers on board the Titanic were men, with a 64,8% male passengers in total passengers.</font>

### 9. Create a chart to show the distribution of passengers by gender (Sex) and whether they survived (Survived) Choose a chart that suits you best, and design it to look clear and presentable.

# Creating a pivot table and retaining the output in a variable named 
pv_survived_gender=df.pivot_table(index='Survived',columns='Sex',values='PassengerId',aggfunc='count')

# View pivot table by calling the variable that we created earlier, pv_survived_gender
pv_survived_gender

### Creating the plot based on the pivot table created earlier
pv_survived_gender.plot.bar(color=['#FF1493','#1E90FF'],
                            width=0.5,
                            figsize=(4,4),
                            title='Passenger distribution by Gender and whether they Survived',
                            xlabel='Survival status',
                            ylabel='No of passengers',
                            stacked=True)

plt.xticks([0,1], labels=["Deceased","Survived"], rotation=45) # using the plt.xticks command the object in chart can be adjusted 

plt.legend(labels=['Female','Male']) # using the plt.legend command the legend can be adjusted

#### <font color=blue> <ins>Conclusion</ins>: Most of the passengers on board the Titanic were men, but the biggest surviving rate was recorded for the female passenger category.</font>

### 10. Use a displot chart to view the age distribution of passengers by age. Follow these guidelines:
• Separate it into two charts according to the values in the gender column

• Within each chart, split the data into series according to the data in the survived column

• Set the display so that the data is stacked

# From this plot we can for sure say that the male passenger were a lot more than the female passenger 
# but the rate of survival is grater for female than male

sns.displot(df, x='Age',col='Sex',hue='Survived',multiple='stack', legend=False)

plt.legend(loc='best', 
           labels=['Survived', 'Deceased'], 
           shadow=True,
           framealpha=1, 
           facecolor='0.94',
           title='')

#### <font color=blue> <ins>Conclusion</ins>: Most of the passengers were young male person, but the highest surviving rate was recorded by the female passenger category.</font>

### 11. Now, examine whether there is a correlation between the different columns.
To do this, select only the columns that have meaning (for example, the
number of clients can vary, but this figure has no effect on the results of the
analysis)
Create a correlation table by applying the corr () method to the data. Save the
result in a variable named corr_mtx
Create a heatmap chart that will run on the correlation table data, and design
it as you wish
Is there a strong positive correlation (values close to 1) or a negative
correlation (values close to 1-) between any of the columns?

# Use the method head() on the Data Frame to have a view on the data
df.head(3)

# Creating the correlation matrix
corr_mtx=df[['Survived','Pclass','Sex','Age','SibSp','Parch','Ticket','Fare']].corr()

# View the output of the correlation matrix.
corr_mtx

# Creating the Heatmap based on the results of the correlation matrix
corr_plt=sns.heatmap(corr_mtx, annot=True, cmap='Blues',linecolor='white',linewidth=1,fmt='.2f')
corr_plt.set_title('Correlation Matrix')

 - #### <font color=blue> <ins>Conclusion</ins>: The correlation matrix summarize the correlations between all numerical variables in the Data Frame.
    
- Positive numbers indicate positive correlations (the two variables move in the same direction), while negative numbers indicate negative correlations (the two variables move in opposite directions).

- The closer the number is to 1 (or -1), the stronger the correlation.

- #### <font color=blue> A correlation coefficient of -0.34 between Survived and Class variables signifies a low negative correlation. 
<font color=blue>That means that the lower the class the higher the chances to survive.</font>
    
- #### <font color=blue> A correlation coefficient of 0.41 between Parch and SibSp variables signifies a low positive correlation. 
<font color=blue>That means that when passengers were traveling with Parents and children the higher the chances were that they had also Siblings and Spouses.</font>
    
- #### <font color=blue> A correlation coefficient of -0.55 between Class and Fare variables signifies a moderate negative correlation. 
<font color=blue>That means that the lower the class the higher the ticket fare.</font>

### Part 2 - Continued Analysis

### 12. Define 3 research questions that will help us find out who has a high chance of surviving, and then answer the questions you have defined.

### 12.1. Is there a correlation between the Passenger gender and the other columns in DataFrame? 
#### Creating a new correlation heatmap by including the gender Series as a numerical value.
#### Checking how strong is the correlation between the Gender and the other variables.

# User defined function that receives gender as parameter and return 1 for 'male' and '2' for 'female'
def gender(sex):
    if sex == 'male':
        return 1
    else:
        return 2

# Apply the above defined function to the 'Sex' series 
# in order to get a new column with the gender information in integer datatype 
df['sex_int'] = df['Sex'].apply(gender)

# Checking the result from applying the function to the 'Sex' series
df[['Sex','sex_int']]

corr_mtx_1=df[['Survived','Pclass','Age','SibSp','Parch','Fare','sex_int']].corr()

corr_mtx_1

# Heatmap
corr_plt_1=sns.heatmap(corr_mtx_1, annot=True, cmap='Blues',linecolor='white',linewidth=1,fmt='.2f')
corr_plt_1.set_title('Correlation Matrix with Gender')

#### <font color=blue> <ins>Conclusion</ins>:
- #### <font color=blue> A correlation coefficient of 0.54 between Sex (gender) and Survived variables signifies a moderate positive correlation.</font> 
<font color=blue>That means that the higher the 'Sex_int' variable, where male=1 and female=2, the higher the surviving chances.</font>

### 12.2. For which class passenger is the survival rate higher?

# Group the data by Class, and save into a new object called g_class
g_class=df.groupby('Pclass')

# size() - method that returns the number of records in each group
g_class.size()

g_class.size().plot.pie(autopct='%1.1f%%',
                                       ylabel='',
                                       title='Passenger distribution by class',
                                       startangle=90,
                                       figsize=(4,4),
                                       explode=(0.1,0.1,0.1))

####  <font color=blue> <ins>Conclusion</ins>: The percentage of passenger in the third class was higher than the percentage of passengers in the first and second class combined. <font>

# Create a pivot table based on the Passenger class and the 'Survived' variable.
# Save the pivot table into a variable pt_class_survived
pt_class_survived=df.pivot_table(index='Pclass',columns='Survived',values='PassengerId',aggfunc='count')

# View the previously created pivot table pt_class_survived
pt_class_survived

# Create plot from the data
pt_class_survived.plot.bar(width=0.8,
                            color=('#FF0066','#0099FF'),
                            edgecolor='black', 
                            figsize=(5,5),
                            title='Survived distribution by class')

plt.xticks(rotation=0) # using the plt.xticks command the object in chart can be adjusted 

plt.legend(labels=['Deceased','Survived']) # using the plt.legend command, the legend can be adjusted

#### <font color=blue> <ins>Conclusion</ins>: The survival chances of a 1st class passenger were higher than a 2nd class and 3rd class passenger.</font>

### 12.3 Passenger age groupping

# User defined function that receives age as parameter and return 'child' for 'Age' <= 11, 'teen' for 'Age' between 12 and 17,
# 'adult' for 'Age' between 18 and 64 and 'senior' for 'Age' > 64.
def age_group(age):
    if age <= 11:
        return 'child'
    if age <= 17:
        return 'teen'
    if age <= 64:
        return 'adult'
    else:
        return 'senior'

# Apply the above defined function to the 'Age' series 
# in order to get a new column with the age category information 
df['age_categ']=df['Age'].apply(age_group)

# Check how many passenger were on boar for each age_category
df['age_categ'].value_counts()

# Check if the function works properly
df.loc[32:45,['Age','age_categ']]

# Create a pivot table based on the Passengers age category and the 'Survived' variable.
# Save the pivot table into a variable pt_agecateg_survived
pt_agecateg_survived=df.pivot_table(index='age_categ',columns='Survived',values='PassengerId',aggfunc='count')

# View the previously created pivot table pt_agecateg_survived
pt_agecateg_survived

# Create plot from the data
pt_agecateg_survived.plot.bar(width=0.8,
                            color=('#FF0066','#0099FF'),
                            edgecolor='black', 
                            figsize=(5,5),
                            title='Survived distribution by age category')

#### <font color=blue> <ins>Conclusion</ins>: Clearly, a higher fraction of children under 12 survived than died. In any other age category, the number of casualties was higher than the number of survivors. </font>

pt_embarked_survived=df.pivot_table(index='Embarked',columns='Survived',values='PassengerId',aggfunc='count')

pt_embarked_survived

# Create plot from the data
pt_embarked_survived.plot.bar(width=0.8,
                            color=('#FF0066','#0099FF'),
                            edgecolor='black', 
                            figsize=(5,5),
                            title='Survived distribution by Embarking port')

#plt.xticks(labels=['Cherbourg','Queenstown','Southampton'], rotation=45)

plt.legend(labels=['Deceased','Survived'])

#C=Cherbourg, Q=Queenstown, S=Southampton

#### <font color=blue> <ins>Conclusion</ins>: 
#### <font color=blue> Portrait of the Titanic passenger: young male, travelling alone, in 3rd class, who boarded from Southampton. </font>

#### <font color=blue> Portrait of the surviving Titanic passenger: young woman travelling with her family or children in first class who boarded from Cherbourg. </font>

 


