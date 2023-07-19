#!/usr/bin/env python
# coding: utf-8

# # Lead Score - Upgrad Case Study Assignment

# ___All the outcomes and understandings are written in <font color= blue> BLUE</font>___

# In[86]:


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')
#Importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # 1 : As the First Step we will Load and Clean the Data
# 
# ##  1.1  Import Data 

# In[87]:


# Loading the data using Pandas
df = pd.read_csv('Leads.csv')
df


# ## 1.2 Inspecting the dataframe
# just to gain a good idea of the dataframes.

# In[88]:


# The .info() code gives almost the entire information that needs to be inspected, so let's start from there
df.info()


# In[89]:


#To get the idea of how the table looks like we can use .head() or .tail() command
df.head()


# In[90]:


# The .shape code gives the no. of rows and columns
df.shape


# In[91]:


#To get an idea of the numeric values, use .describe()
df.describe()


# ## 1.3 Cleaning the dataframe

# In[92]:


# Converting all the values to lower case
df = df.applymap(lambda s:s.lower() if type(s) == str else s)


# In[93]:


# Replacing 'Select' with NaN (Since it means no option is selected)
df = df.replace('select',np.nan)


# In[94]:


# Checking if there are columns with one unique value since it won't affect our analysis
df.nunique()


# In[95]:


# Dropping unique valued columns
df1= df.drop(['Magazine','Receive More Updates About Our Courses','I agree to pay the amount through cheque','Get updates on DM Content','Update me on Supply Chain Content'],axis=1)


# In[96]:


# Checking the percentage of missing values
round(100*(df1.isnull().sum()/len(df1.index)), 2)


# In[97]:


# Removing all the columns that are no required and have 35% null values
df2 = df1.drop(['Asymmetrique Profile Index','Asymmetrique Activity Index','Asymmetrique Activity Score','Asymmetrique Profile Score','Lead Profile','Tags','Lead Quality','How did you hear about X Education','City','Lead Number'],axis=1)
df2.head()


# In[98]:


# Rechecking the percentage of missing values
round(100*(df2.isnull().sum()/len(df2.index)), 2)


# <font color= blue>___In the dataset, four columns contain a significant number of null variables. However, discarding rows with missing values is not a viable option as it would result in a considerable loss of important data, especially since these columns are crucial. Instead, we have decided to handle the null values by replacing them with the label 'not provided'. This approach ensures that we retain all available data while minimizing the presence of null values. Should any issues arise during the model's implementation, we can easily identify and exclude these instances, thereby maintaining the integrity of the analysis..___</font>

# In[99]:


df2['Specialization'] = df2['Specialization'].fillna('not provided') 
df2['What matters most to you in choosing a course'] = df2['What matters most to you in choosing a course'].fillna('not provided')
df2['Country'] = df2['Country'].fillna('not provided')
df2['What is your current occupation'] = df2['What is your current occupation'].fillna('not provided')
df2.info()


# In[100]:


# Rechecking the percentage of missing values
round(100*(df2.isnull().sum()/len(df2.index)), 2)


# In[101]:


df2["Country"].value_counts()


# In[102]:


def slots(x):
    category = ""
    if x == "india":
        category = "india"
    elif x == "not provided":
        category = "not provided"
    else:
        category = "outside india"
    return category

df2['Country'] = df2.apply(lambda x:slots(x['Country']), axis = 1)
df2['Country'].value_counts()


# In[103]:


# Rechecking the percentage of missing values
round(100*(df2.isnull().sum()/len(df2.index)), 2)


# In[104]:


# Checking the percent of lose if the null values are removed
round(100*(sum(df2.isnull().sum(axis=1) > 1)/df2.shape[0]),2)


# In[105]:


df3 = df2[df2.isnull().sum(axis=1) <1]


# In[106]:


# Code for checking number of rows left in percent
round(100*(df3.shape[0])/(df.shape[0]),2)


# In[107]:


# Rechecking the percentage of missing values
round(100*(df3.isnull().sum()/len(df3.index)), 2)


# In[108]:


# To familiarize all the categorical values
for column in df3:
    print(df3[column].astype('category').value_counts())
    print('----------------------------------------------------------------------------------------')


# In[109]:


# Removing Id values since they are unique for everyone
df_final = df3.drop('Prospect ID',1)
df_final.shape


# ## 2. EDA

# ### 2.1. Univariate Analysis

# #### 2.1.1. Categorical Variables

# In[110]:


df_final.info()


# In[111]:


plt.figure(figsize = (20,40))

plt.subplot(6,2,1)
sns.countplot(df_final['Lead Origin'])
plt.title('Lead Origin')

plt.subplot(6,2,2)
sns.countplot(df_final['Do Not Email'])
plt.title('Do Not Email')

plt.subplot(6,2,3)
sns.countplot(df_final['Do Not Call'])
plt.title('Do Not Call')

plt.subplot(6,2,4)
sns.countplot(df_final['Country'])
plt.title('Country')

plt.subplot(6,2,5)
sns.countplot(df_final['Search'])
plt.title('Search')

plt.subplot(6,2,6)
sns.countplot(df_final['Newspaper Article'])
plt.title('Newspaper Article')

plt.subplot(6,2,7)
sns.countplot(df_final['X Education Forums'])
plt.title('X Education Forums')

plt.subplot(6,2,8)
sns.countplot(df_final['Newspaper'])
plt.title('Newspaper')

plt.subplot(6,2,9)
sns.countplot(df_final['Digital Advertisement'])
plt.title('Digital Advertisement')

plt.subplot(6,2,10)
sns.countplot(df_final['Through Recommendations'])
plt.title('Through Recommendations')

plt.subplot(6,2,11)
sns.countplot(df_final['A free copy of Mastering The Interview'])
plt.title('A free copy of Mastering The Interview')

plt.subplot(6,2,12)
sns.countplot(df_final['Last Notable Activity']).tick_params(axis='x', rotation = 90)
plt.title('Last Notable Activity')


plt.show()


# In[112]:


sns.countplot(df_final['Lead Source']).tick_params(axis='x', rotation = 90)
plt.title('Lead Source')
plt.show()


# In[113]:


plt.figure(figsize = (20,30))
plt.subplot(2,2,1)
sns.countplot(df_final['Specialization'],palette ="tab10").tick_params(axis='x', rotation = 90)
plt.title('Specialization')
plt.subplot(2,2,2)
sns.countplot(df_final['What is your current occupation'],palette ="viridis").tick_params(axis='x', rotation = 90)
plt.title('Current Occupation')
plt.subplot(2,2,3)
sns.countplot(df_final['What matters most to you in choosing a course'],palette ="viridis").tick_params(axis='x', rotation = 90)
plt.title('What matters most to you in choosing a course')
plt.subplot(2,2,4)
sns.countplot(df_final['Last Activity'],palette ="Paired").tick_params(axis='x', rotation = 90)
plt.title('Last Activity')
plt.show()


# In[114]:


sns.countplot(df['Converted'])
plt.title('Converted("Y variable")')
plt.show()


# #### 2.1.1. Numerical Variables

# In[115]:


df_final.info()


# In[116]:


plt.figure(figsize = (10,10))
plt.subplot(221)
plt.hist(df_final['TotalVisits'], bins = 200,color = "skyblue")
plt.title('Total Visits')
plt.xlim(0,25)

plt.subplot(222)
plt.hist(df_final['Total Time Spent on Website'], bins = 10,color = "green")
plt.title('Total Time Spent on Website')

plt.subplot(223)
plt.hist(df_final['Page Views Per Visit'], bins = 20,color = "purple")
plt.title('Page Views Per Visit')
plt.xlim(0,20)
plt.show()


# ### 2.1. Relating all the categorical variables to Converted

# In[117]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Lead Origin', hue='Converted', data= df_final,palette ="YlOrBr").tick_params(axis='x', rotation = 90)
plt.title('Lead Origin')

plt.subplot(1,2,2)
sns.countplot(x='Lead Source', hue='Converted', data= df_final,palette ="YlOrBr").tick_params(axis='x', rotation = 90)
plt.title('Lead Source')
plt.show()


# In[118]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Do Not Email', hue='Converted', data= df_final,palette ="icefire").tick_params(axis='x', rotation = 90)
plt.title('Do Not Email')

plt.subplot(1,2,2)
sns.countplot(x='Do Not Call', hue='Converted', data= df_final,palette ="icefire").tick_params(axis='x', rotation = 90)
plt.title('Do Not Call')
plt.show()


# In[119]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Last Activity', hue='Converted', data= df_final,palette ="cubehelix").tick_params(axis='x', rotation = 90)
plt.title('Last Activity')

plt.subplot(1,2,2)
sns.countplot(x='Country', hue='Converted', data= df_final,palette ="cubehelix").tick_params(axis='x', rotation = 90)
plt.title('Country')
plt.show()


# In[120]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Specialization', hue='Converted', data= df_final,palette ="viridis").tick_params(axis='x', rotation = 90)
plt.title('Specialization')

plt.subplot(1,2,2)
sns.countplot(x='What is your current occupation', hue='Converted', data= df_final,palette ="viridis").tick_params(axis='x', rotation = 90)
plt.title('What is your current occupation')
plt.show()


# In[121]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='What matters most to you in choosing a course', hue='Converted', data= df_final,palette ="rocket").tick_params(axis='x', rotation = 90)
plt.title('What matters most to you in choosing a course')

plt.subplot(1,2,2)
sns.countplot(x='Search', hue='Converted', data= df_final,palette ="rocket").tick_params(axis='x', rotation = 90)
plt.title('Search')
plt.show()


# In[122]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Newspaper Article', hue='Converted', data= df_final,palette ="Paired").tick_params(axis='x', rotation = 90)
plt.title('Newspaper Article')

plt.subplot(1,2,2)
sns.countplot(x='X Education Forums', hue='Converted', data= df_final,palette ="Paired").tick_params(axis='x', rotation = 90)
plt.title('X Education Forums')
plt.show()


# In[123]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Newspaper', hue='Converted', data= df_final,palette ="hls").tick_params(axis='x', rotation = 90)
plt.title('Newspaper')

plt.subplot(1,2,2)
sns.countplot(x='Digital Advertisement', hue='Converted', data= df_final,palette ="hls").tick_params(axis='x', rotation = 90)
plt.title('Digital Advertisement')
plt.show()


# In[124]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Through Recommendations', hue='Converted', data= df_final, palette ="Set1").tick_params(axis='x', rotation = 90)
plt.title('Through Recommendations')

plt.subplot(1,2,2)
sns.countplot(x='A free copy of Mastering The Interview', hue='Converted', data= df_final, palette ="Set1").tick_params(axis='x', rotation = 90)
plt.title('A free copy of Mastering The Interview')
plt.show()


# In[125]:


sns.countplot(x='Last Notable Activity', hue='Converted', data= df_final, palette = "Set2").tick_params(axis='x', rotation = 90)
plt.title('Last Notable Activity')
plt.show()


# In[126]:


# To check the correlation among varibles
plt.figure(figsize=(10,5))
sns.heatmap(df_final.corr(),cmap="BuPu")
plt.show()


# <font color= blue>___Based on the above Exploratory Data Analysis (EDA), it is evident that several elements in the dataset contain a small amount of data. Consequently, these elements are likely to have limited relevance to our analysis. Therefore, during the data analysis and modeling process, we may consider excluding these elements to focus on the more substantial and informative aspects of the dataset, enhancing the overall quality and accuracy of our results.___</font>

# In[127]:


numeric = df_final[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]
numeric.describe(percentiles=[0.25,0.5,0.75,0.9,0.99])


# <font color= blue>___moving on to analysis due to lack of outliers___</font>

# ## 3. Dummy Variables

# In[128]:


df_final.info()


# In[129]:


df_final.loc[:, df_final.dtypes == 'object'].columns


# In[130]:


# Create dummy variables using the 'get_dummies'
dummy = pd.get_dummies(df_final[['Lead Origin','Specialization' ,'Lead Source', 'Do Not Email', 'Last Activity', 'What is your current occupation','A free copy of Mastering The Interview', 'Last Notable Activity']], drop_first=True)
# Add the results to the master dataframe
df_final_dum = pd.concat([df_final, dummy], axis=1)
df_final_dum


# In[131]:


df_final_dum = df_final_dum.drop(['What is your current occupation_not provided','Lead Origin', 'Lead Source', 'Do Not Email', 'Do Not Call','Last Activity', 'Country', 'Specialization', 'Specialization_not provided','What is your current occupation','What matters most to you in choosing a course', 'Search','Newspaper Article', 'X Education Forums', 'Newspaper','Digital Advertisement', 'Through Recommendations','A free copy of Mastering The Interview', 'Last Notable Activity'], 1)
df_final_dum


# ## 4. Test-Train Split

# In[132]:


# Import the required library
from sklearn.model_selection import train_test_split


# In[133]:


X = df_final_dum.drop(['Converted'], 1)
X.head()


# In[134]:


# Putting the target variable in y
y = df_final_dum['Converted']
y.head()


# In[135]:


# Split the dataset into 70% and 30% for train and test respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=10)


# In[136]:


# Import MinMax scaler
from sklearn.preprocessing import MinMaxScaler
# Scale the three numeric features
scaler = MinMaxScaler()
X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])
X_train.head()


# In[137]:


# To check the correlation among varibles
plt.figure(figsize=(30,50))
sns.heatmap(X_train.corr(),cmap="crest",linewidth=.5)
plt.show()


# <font color= blue>___We have decided to drop variables after RFE___</font>

# ## 5. Building The Model

# In[138]:


# Import 'LogisticRegression'
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[139]:


# Import RFE
from sklearn.feature_selection import RFE


# In[140]:


# Running RFE with 15 variables as output
rfe = RFE(logreg, n_features_to_select=15)
rfe = rfe.fit(X_train, y_train)


# In[141]:


# Features that have been selected by RFE
list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[142]:


# Put all the columns selected by RFE in the variable 'col'
col = X_train.columns[rfe.support_]


# <font color= blue>___Once the variables are selected using Recursive Feature Elimination (RFE), the next step is to assess their statistical significance using p-values and calculate their Variance Inflation Factors (VIFs) to evaluate multicollinearity. P-values: After selecting the variables with RFE, you can perform a statistical analysis such as a regression model or a hypothesis test for each selected variable. The p-value associated with each variable indicates the probability of observing the relationship between the variable and the outcome by chance alone. Generally, variables with low p-values (typically below a significance level, e.g., 0.05) are considered statistically significant and are more likely to have a meaningful impact on the outcome. Variance Inflation Factors (VIFs): Multicollinearity occurs when there is a high correlation among predictor variables in a regression model. VIF is a measure used to assess the severity of multicollinearity. It quantifies how much the variance of the estimated regression coefficient is inflated due to multicollinearity. Generally, a VIF value greater than 1 indicates the presence of multicollinearity, and values above a certain threshold (e.g., 5 or 10) are considered problematic.___</font>

# In[143]:


# Selecting columns selected by RFE
X_train = X_train[col]


# In[144]:


# Importing statsmodels
import statsmodels.api as sm


# In[145]:


X_train_sm = sm.add_constant(X_train)
logm1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[146]:


# Importing 'variance_inflation_factor'
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[147]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# <font color= blue>___If the VIF values for the variables are acceptable, indicating no significant multicollinearity, but some of the p-values are not within an acceptable range (e.g., greater than 0.05), it suggests that certain variables may not be statistically significant and could be adversely affecting the model's performance.In this case, it is appropriate to consider removing the variable 'Last Notable Activity had a phone conversation' from the analysis. This variable does not seem to have a meaningful impact on the outcome, as suggested by its high p-value. By removing this variable, you can potentially improve the model's performance and focus on the more relevant and statistically significant predictors.'___</font>

# In[148]:


X_train.drop('Last Notable Activity_had a phone conversation', axis = 1, inplace = True)


# In[149]:


# Refit the model with the new set of features
X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[150]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# <font color= blue>___Removing 'What is your current occupation housewife' due to same above mentioned reason___</font>

# In[151]:


X_train.drop('What is your current occupation_housewife', axis = 1, inplace = True)


# In[152]:


# Refit the model with the new set of features
X_train_sm = sm.add_constant(X_train)
logm3 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[153]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# <font color= blue>___Removing the 'What is your current occupation other'___</font>

# In[154]:


X_train.drop('What is your current occupation_other', axis = 1, inplace = True)


# In[155]:


# Refit the model with the new set of features
X_train_sm = sm.add_constant(X_train)
logm4 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[156]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# <font color= blue>___As we can see that the VIF values are good and the p-values are below 0.05. So we can proceed with the model.___</font>

# ## 6. Trying to make Prediction

# In[157]:


# Predicting the probabilities on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[158]:


# Reshaping to an array
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[159]:


# Data frame with given convertion rate and probablity of predicted ones
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final.head()


# In[160]:


# Substituting 0 or 1 with the cut off as 0.5
y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# ## 7. Evaluating the Model

# In[161]:


# Importing metrics from sklearn for evaluation
from sklearn import metrics


# In[162]:


# Creating confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# In[163]:


# Predicted     not_churn    churn
# Actual
# not_churn        3403       492
# churn             729      1727


# In[164]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)


# <font color= blue>___accuracy around 81% seems to be a very good value___</font>

# In[165]:


# Substituting the value of true positive
TP = confusion[1,1]
# Substituting the value of true negatives
TN = confusion[0,0]
# Substituting the value of false positives
FP = confusion[0,1] 
# Substituting the value of false negatives
FN = confusion[1,0]


# In[166]:


# Calculating the sensitivity
TP/(TP+FN)


# In[167]:


# Calculating the specificity
TN/(TN+FP)


# <font color= blue>___With a cutoff of 0.5, your model is achieving an overall accuracy of approximately 81%. This means that 81% of the predictions made by the model are correct.The sensitivity of around 70% indicates that the model correctly identifies around 70% of the positive cases (true positives) out of all the actual positive cases. It is also known as the true positive rate or recall.The specificity of around 87% indicates that the model correctly identifies around 87% of the negative cases (true negatives) out of all the actual negative cases. It is also known as the true negative rate.It's important to note that accuracy, sensitivity, and specificity are essential evaluation metrics, but their relative importance depends on the specific context of your problem. For instance, if false negatives (missed positive cases) are critical and need to be minimized, you may focus on improving sensitivity even if it comes at the expense of some decrease in specificity.Similarly, if false positives (false alarms) are problematic, increasing specificity might be a priority. It's essential to consider the trade-offs between these metrics based on the specific goals and requirements of your model's application. Additionally, exploring other evaluation metrics like precision, F1-score, or ROC-AUC can provide a more comprehensive understanding of your model's performance.___</font>

# ## 7. Further Optimising the Cut off (ROC Curve)

# The previous cut off was randomely selected. Now to find the optimum one

# In[168]:


# ROC function
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[169]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, drop_intermediate = False )


# In[170]:


# Call the ROC function
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# <font color= blue>___An AUC-ROC value of 0.87 suggests that your model is performing well in distinguishing between positive and negative cases, and it has a good level of discriminatory power. However, as always, it's essential to consider the specific context of your problem and the desired performance threshold for the task at hand.___</font>

# In[171]:


# Creating columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[172]:


# Creating a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
# Making confusing matrix to find values of sensitivity, accurace and specificity for each level of probablity
from sklearn.metrics import confusion_matrix
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
cutoff_df


# In[173]:


# Plotting it
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# <font color= blue>___the graph shows that the optimal cut off is at 0.35.___</font>

# In[174]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.35 else 0)
y_train_pred_final.head()


# In[175]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[176]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[177]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[178]:


# Calculating the sensitivity
TP/(TP+FN)


# In[179]:


# Calculating the specificity
TN/(TN+FP)


# <font color= blue>___With a cutoff of 0.35, your model is achieving an overall accuracy of around 80%. This means that 80% of the predictions made by the model are correct.The sensitivity at around 80% indicates that the model correctly identifies around 80% of the positive cases (true positives) out of all the actual positive cases. It is also known as the true positive rate or recall.The specificity at around 80% indicates that the model correctly identifies around 80% of the negative cases (true negatives) out of all the actual negative cases. It is also known as the true negative rate.___</font>

# ## 8. Making the Predictions on Test set

# In[180]:


# Scaling numeric values
X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.transform(X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])


# In[181]:


# Substituting all the columns in the final train model
col = X_train.columns


# In[182]:


# Select the columns in X_train for X_test as well
X_test = X_test[col]
# Add a constant to X_test
X_test_sm = sm.add_constant(X_test[col])
X_test_sm
X_test_sm


# In[183]:


# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)
# Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)
# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[184]:


# Making prediction using cut off 0.35
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.35 else 0)
y_pred_final


# In[185]:


# Check the overall accuracy
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[186]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[187]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[188]:


# Calculating the sensitivity
TP/(TP+FN)


# In[189]:


# Calculating the specificity
TN/(TN+FP)


# <font color= blue>___As we can see,the current cut off as 0.35 we have accuracy, sensitivity and specificity of around 80%.___</font>

# ## 9. Precision-Recall

# In[190]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# In[191]:


# Precision = TP / TP + FP
confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[192]:


#Recall = TP / TP + FN
confusion[1,1]/(confusion[1,0]+confusion[1,1])


# <font color= blue>___As we can see, current cut off is 0.35 we have Precision around 78% and Recall around 70%___</font>

# ### 9.1. Precision and recall tradeoff

# In[193]:


from sklearn.metrics import precision_recall_curve


# In[194]:


y_train_pred_final.Converted, y_train_pred_final.Predicted


# In[195]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# In[196]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[197]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.41 else 0)
y_train_pred_final.head()


# In[198]:


# Accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[199]:


# Creating confusion matrix again
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[200]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[201]:


# Precision = TP / TP + FP
TP / (TP + FP)


# In[202]:


#Recall = TP / TP + FN
TP / (TP + FN)


# <font color= blue>___As we can see, current cut off as 0.41 we have Precision around 74% and Recall around 76%___</font>

# ## 10. Making the Prediction on Test set

# In[203]:


# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)
# Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)
# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[204]:


# Making prediction using cut off 0.41
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.41 else 0)
y_pred_final


# In[205]:


# Check the overall accuracy
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[206]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[207]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[208]:


# Precision = TP / TP + FP
TP / (TP + FP)


# In[209]:


#Recall = TP / TP + FN
TP / (TP + FN)


# <font color= blue>___As we can see, the current cut off as 0.41 we have Precision around 73% and Recall around 76%___</font>

# ## We can conclude
# Based on the analysis, it has been found that certain variables play a crucial role in determining potential buyers for X Education. In descending order of importance, these variables are:
# 
# Total time spent on the Website.
# 
# Total number of visits.
# Lead source:
# a. Google
# b. Direct traffic
# c. Organic search
# d. Welingak website
# 
# Last activity:
# a. SMS
# b. Olark chat conversation
# Lead origin as Lead add format.
# Current occupation as a working professional.
# These variables have been identified as significant predictors in influencing potential buyers' decisions to change their minds and purchase X Education's courses. By leveraging this knowledge, X Education can optimize their marketing strategies, tailor their communication, and target potential buyers effectively. With a higher chance of reaching potential buyers, X Education can flourish and enhance their conversion rates, leading to increased success and growth for the organization.

# In[ ]:




