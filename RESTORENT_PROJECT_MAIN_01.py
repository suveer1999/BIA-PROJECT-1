## Problem Statement

import numpy as np
import pandas as pd

df = pd.read_csv('zomatoData.csv')

df.head()

df.shape

df.info()

df.isnull().sum()

df.loc[:, df.columns[df.isnull().any()]].isnull().sum().value_counts().sum()

df.loc[:, df.columns[df.isnull().any()]].isnull().sum()

list(df.columns)

df.head(1)

df.rest_type.value_counts()

df.cuisines.value_counts()

df['listed_in(type)'].value_counts()

# Adjust display options
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

df['listed_in(city)'].value_counts().count()

df['location'].value_counts().count()

df.head(10)

df['listed_in(city)'].value_counts().sum()

df['location'].value_counts().sum()

df.rest_type.value_counts()

df.cuisines.value_counts()

df.reviews_list

# Taking Necessary Columns

df_rest = df[['online_order', 'book_table', 'rate', 'votes', 'rest_type', 'listed_in(type)', 'listed_in(city)']]
df_rest.head()

df_rest.shape

df_rest.isnull().sum()

df_rest.rate

df_rest.dtypes

df_rest.rate

for col in df_rest.rate:
    print(type(col))


def handle(value):
    if value == 'NEW' or value == '-':
        return np.nan
    else:
        return float(str(value).replace('/5', ''))


df_rest.loc[:, 'rate'] = df_rest['rate'].apply(handle)
df_rest['rate'].unique()

df_rest.rate.isnull().sum()

df_rest.rate.fillna(df_rest.rate.mean(), inplace=True)

df_rest.loc[:, 'rate'] = df_rest.rate.round(1)

df_rest.rate

df_rest.rate.mean()

df_rest.rate.isnull().sum()

df_rest.rest_type.isnull().sum()

df_rest.rest_type

df_rest.dropna(inplace=True)

df_rest.isnull().sum()

df_rest.shape

df_rest.columns

df['approx_cost(for two people)'].isnull().sum()

df_rest.loc[:, 'price'] = df['approx_cost(for two people)']

df_rest.columns

df_rest.isnull().sum()

df_rest.price.unique()


def price_handle(value):
    if ',' in str(value):
        return float(value.replace(',', ''))
    return float(value)


df_rest.loc[:, 'price'] = df_rest.price.apply(price_handle)
df_rest.price.unique()

df_rest.price.dtypes

df_rest.price.isnull().sum()

df_rest.price = pd.to_numeric(df_rest.price)

df_rest.price.dtypes

df_rest.info()

df_rest.price.isnull().sum()

df_rest.price.mean()

import warnings

warnings.filterwarnings('ignore')

df_rest.price.fillna(df_rest.price.mean(), inplace=True)

df_rest.price.isnull().sum()

df_rest.price.mean()

df_rest.info()

df_rest.isnull().sum()

df_rest.head()

df.cuisines

df.cuisines.value_counts()

df.cuisines.shape

df.cuisines.isnull().sum()

df.isnull().sum()

df_rest.shape

df_rest['cuisines'] = df.cuisines

df_rest.shape

df_rest.columns

df_rest.shape

df_rest.isnull().sum()

df_rest.dropna(inplace=True)

df_rest.shape

df_rest.head(20)

df_rest.cuisines.value_counts()

df_rest.cuisines.unique()

df_rest.cuisines.nunique()

df_rest.rest_type.value_counts()

df_rest.rest_type.unique()

df_rest.rest_type.nunique()

df_rest.rest_type.value_counts()

rest_type_less_1000 = pd.DataFrame(df_rest['rest_type'].value_counts())
rest_type_less_1000 = rest_type_less_1000[rest_type_less_1000['count'] < 1000]
rest_type_less_1000


def type(val):
    if (val in rest_type_less_1000.index):
        return 'Others'
    else:
        return val


df_rest['rest_type'] = df_rest['rest_type'].apply(type)
df_rest['rest_type']

df_rest['rest_type'].unique()


def rest(val):
    if val == 'Takeaway, Delivery':
        return 'Delivery'
    elif val == 'Casual Dining, Bar':
        return 'Bar'
    else:
        return val


df_rest['rest_type'] = df_rest['rest_type'].apply(rest)
df_rest['rest_type'].unique()

df_rest.cuisines.value_counts()


def cleaning_cuisine(val):
    if (val == 'North Indian, Chinese'):
        return 'Multicuisine'
    elif (val == 'Bakery, Desserts'):
        return 'Desserts'
    elif (val == 'Biryani'):
        return 'Multicuisine'
    elif ('Pizza' in str(val)):
        return 'Italian'
    elif (val == 'North Indian'):
        return 'North Indian'
    elif (val == 'South Indian'):
        return 'South Indian'
    else:
        return 'Others'


df_rest['cuisines'] = df_rest['cuisines'].apply(cleaning_cuisine)

df_rest['cuisines'].value_counts()

df_rest['cuisines'].unique()

df_rest[df_rest.duplicated()]

df_rest[
    (df_rest['online_order'] == 'No') & (df_rest['book_table'] == 'No') & (df_rest['rest_type'] == 'Dessert Parlor') & (
                df_rest['price'] == 150)]

df_rest[df_rest.duplicated()].shape

df_rest.drop_duplicates(inplace=True)

df_rest.shape

df_rest.head(10)

df_rest.isna().sum()

df_rest.info()

df_rest.shape

df_rest.describe()

df_cat = df_rest.select_dtypes(include='object')
df_num = df_rest.select_dtypes(include=[np.number])

df_cat.dtypes

df_num.dtypes

import matplotlib.pyplot as plt
import seaborn as sns

for col in df_num:
    plt.style.use('ggplot')
    plt.title('Outlier dedection of {} using Boxplot'.format(col))
    sns.boxplot(df_rest[col])
    plt.show()

for col in df_num:
    plt.style.use('ggplot')
    plt.title('Outlier dedection of {} using Boxplot'.format(col))
    sns.boxplot(np.log(df_rest[col]))
    plt.show()

sns.histplot(x='rate', data=df_num, kde=True)
plt.show()

df_num['rate'].describe()

df_num['rate'].value_counts()

sns.histplot(x=np.log(df_num['rate']), data=df_num, kde=True)
plt.show()

df_num['votes'].describe()

df_num['votes'].value_counts(ascending=False)

df_rest.head()

sns.kdeplot(x='votes', data=df_num)
plt.show()

df_num['price'].describe()

df_num['price'].value_counts().count()

df_num['price'].nunique()

df_num['price'].unique()

df_rest['price'] = round(df_rest['price'], 0)

df_num['price'] = round(df_num['price'], 0)

df_num['price'].unique()

df_num['price'].nunique()

df_num['votes'].value_counts(ascending=False)

df_num['votes'].index

df_num['price'].value_counts()

for col in df_num:
    plt.style.use('ggplot')
    plt.title('Outlier dedection of {} using Boxplot'.format(col))
    sns.boxplot(df_rest[col])
    plt.show()

# Exploratory Data Analysis

df_rest.head()

df_rest['rest_type'].value_counts()

sns.set_style('darkgrid')
sns.countplot(x=df_rest['rest_type'])
plt.xticks(rotation=90)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.countplot(x=df_rest['listed_in(type)'], ax=axes[0])
axes[0].set_title("Count of types of restaurant")
axes[0].set_xticklabels(df_rest['listed_in(type)'].value_counts().index, rotation=90)
sns.countplot(x=df_rest['listed_in(city)'], ax=axes[1])
axes[1].set_title("Count of number of restaurants in each city")
plt.xticks(rotation=90)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=df_rest['online_order'], ax=axes[0])
axes[0].set_title("Count of restaurants which has online delivery")
sns.countplot(x=df_rest['book_table'], ax=axes[1])
axes[1].set_title("Count of restaurants which allows booking table")
plt.show()

sns.countplot(x=df['listed_in(city)'], hue=df['book_table'])
plt.xticks(rotation=90)
plt.show()

sns.countplot(x=df['listed_in(city)'], hue=df['online_order'])
plt.xticks(rotation=90)
plt.show()

price_df = df_rest.groupby('rest_type')['price'].median().reset_index()
order_list = price_df.sort_values(by='price', ascending=False)['rest_type'].to_list()
sns.barplot(x=price_df['rest_type'], y=price_df['price'], order=order_list)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(14, 8))
sns.barplot(x="rest_type", y="rate", data=df_rest)
plt.title("We are analysing which type of Restaurants have Maximum Avearge Rating")
plt.show()

df_num.head()

df_num.corr()['price']

df_cat.head()

df_rest[df_rest['rest_type'] == 'Quick Bites']['price'].to_list()

sns.distplot(x=df_rest['rate'], kde=False, bins=3)
plt.show()

sns.kdeplot(x="rate", data=df_rest, hue="votes", legend=False)
plt.show()

plt.scatter(df_rest['rate'], df_rest['votes'])
plt.xlabel('Ratings')
plt.ylabel('Votes')
plt.title('Scatter Plot of Ratings vs Votes')
plt.show()

sns.kdeplot(x="votes", data=df_rest)
plt.show()

df_num.corr()['votes']

sns.pairplot(df_rest, vars=['votes', 'rate'], hue='rate')
plt.show()

pd.cut(df_rest['rate'], ordered=True, bins=3, precision=1).value_counts()

df_rest.shape

df_rest.groupby(['rate'])['votes'].sum()

plt.scatter(df_rest['rate'], df_rest['votes'])
plt.xlabel('Ratings')
plt.ylabel('Votes')
plt.title('Scatter Plot of Ratings vs Votes')
plt.show()

pd.cut(df_rest['rate'], ordered=True, bins=3, precision=1).value_counts()

len(list(df_rest['votes']))

df_rest['votes'].max()

df_rest['rest_type'].value_counts().sum()

df_rest['rate'].unique()

sns.distplot(x=df_rest['votes'].unique(), kde=False, bins=4)
plt.show()

sns.distplot(x=df_rest['rate'], kde=True, bins=3)
plt.show()

df_rest.groupby('rate')['votes'].sum().sort_values(ascending=False).reset_index()

df_rest['votes'].sum()

hist, bin_edges = np.histogram(df_rest['votes'], bins=4)

hist

list(bin_edges)

categories = ['4th class', '3rd class', '2nd class', '1st class']
rate_cat = pd.cut(df_rest['votes'], bins=bin_edges, labels=categories)
print(rate_cat)

df_rest['votes']

rate_cat.isnull().sum()

df_rest[df_rest['votes'] == 0]['votes'].count()

rate_cat.fillna('4th class', inplace=True)

df_rest['rate_cat'] = rate_cat

df_rest.sort_values(by='votes', ascending=False).head(200)

df_pro = df_rest.drop(['rate', 'votes'], axis=1)

df_pro.head()

df_pro.isnull().sum()

df_pro.info()

df_pro['rate_cat'] = df_pro['rate_cat'].astype('object')

df_pro.info()

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Assuming 'df_pro' is your DataFrame containing the data
# Extract the columns for the variables you want to analyze
data = df_pro[['online_order', 'book_table', 'rate_cat', 'price']]

# Fit the ANOVA model
model = ols('price ~ C(online_order) * C(book_table) * C(rate_cat)', data=data).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

# Extract F-test statistic and p-value
f_statistic = anova_table['F'][0]  # F-test statistic is the value in the first row of the 'F' column
p_value = anova_table['PR(>F)'][0]  # p-value is the value in the first row of the 'PR(>F)' column

# Set the significance level (alpha)
alpha = 0.05

# Print the results
print("F-test statistic:", f_statistic)
print("p-value:", p_value)

# Interpret the results
if p_value < alpha:
    print(
        "Reject the null hypothesis: There is a significant difference in price based on online_order, book_table, and rate_cat.")
else:
    print(
        "Accept the null hypothesis: There is no significant difference in price based on online_order, book_table, and rate_cat.")

df_rest['rate_cat'] = df_pro['rate_cat']

df_rest.head()

df_cat = df_rest.select_dtypes(include='object')
df_num = df_rest.select_dtypes(include=np.number)

df_cat.head()

df_num.head()

categorical_columns = list(df_cat.columns)

# dummy_var = pd.get_dummies(data = df_cat, drop_first = True, dtype = 'int8')
# dummy_var.head()

dummy_var = pd.get_dummies(data=df_cat, prefix=None, prefix_sep='_', columns=categorical_columns, drop_first=True,
                           dtype='int8')

df_target = df_num['price']

df_num.drop(columns='price', axis=1, inplace=True)

df_num.columns

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# initialize the standard scalar
X_scaler = StandardScaler()

# scale all the numeric variables
# standardize all the columns of the dataframe 'df_num'
num_scaled = X_scaler.fit_transform(df_num)

# create a dataframe of scaled numerical variables
# pass the required column names to the parameter 'columns'
df_num_scaled = pd.DataFrame(num_scaled, columns=df_num.columns)

# standardize the target variable explicitly and store it in a new variable 'y'
y = (df_target - df_target.mean()) / df_target.std()

y

# X = pd.concat([df_num_scaled, dummy_var], axis = 1)

# Merge DataFrames on index
# X = df_num_scaled.merge(dummy_var, left_index=True, right_index=True)

# Reset indices of both DataFrames
df_num_scaled_reset = df_num_scaled.reset_index(drop=True)
dummy_var_reset = dummy_var.reset_index(drop=True)

# Concatenate the DataFrames
X = pd.concat([df_num_scaled_reset, dummy_var_reset], axis=1)

X.shape

# 'metrics' from sklearn is used for evaluating the model performance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# split data into train subset and test subset
# set 'random_state' to generate the same dataset each time you run the code
# 'test_size' returns the proportion of data to be included in the testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.2)

# check the dimensions of the train & test subset using 'shape'
# print dimension of train set
print('X_train', X_train.shape)
print('y_train', y_train.shape)

# print dimension of test set
print('X_test', X_test.shape)
print('y_test', y_test.shape)


# create a generalized function to calculate the RMSE values for train set
def get_train_rmse(model):
    # For training set:
    # train_pred: prediction made by the model on the training dataset 'X_train'
    # y_train: actual values ofthe target variable for the train dataset

    # predict the output of the target variable from the train data
    train_pred = model.predict(X_train)

    # calculate the MSE using the "mean_squared_error" function

    # MSE for the train data
    mse_train = mean_squared_error(y_train, train_pred)

    # take the square root of the MSE to calculate the RMSE
    # round the value upto 4 digits using 'round()'
    rmse_train = round(np.sqrt(mse_train), 4)

    # return the training RMSE
    return (rmse_train)


# create a generalized function to calculate the RMSE values test set
def get_test_rmse(model):
    # For testing set:
    # test_pred: prediction made by the model on the test dataset 'X_test'
    # y_test: actual values of the target variable for the test dataset

    # predict the output of the target variable from the test data
    test_pred = model.predict(X_test)

    # MSE for the test data
    mse_test = mean_squared_error(y_test, test_pred)

    # take the square root of the MSE to calculate the RMSE
    # round the value upto 4 digits using 'round()'
    rmse_test = round(np.sqrt(mse_test), 4)

    # return the test RMSE
    return (rmse_test)


# define a function to get R-squared and adjusted R-squared value
def get_score(model):
    # score() returns the R-squared value
    r_sq = model.score(X_train, y_train)

    # return the R-squared and adjusted R-squared value
    return (r_sq)


# initiate linear regression model
linreg = LinearRegression()

# build the model using X_train and y_train
# use fit() to fit the regression model
MLR_model = linreg.fit(X_train, y_train)

# print the R-squared value for the model
# score() returns the R-squared value
MLR_model.score(X_train, y_train)

r_score = get_score(linreg)
r_score

# print training RMSE
print('RMSE on train set: ', get_train_rmse(MLR_model))

# print training RMSE
print('RMSE on test set: ', get_test_rmse(MLR_model))

# calculate the difference between train and test set RMSE
difference = abs(get_test_rmse(MLR_model) - get_train_rmse(MLR_model))

# print the difference between train and test set RMSE
print('Difference between RMSE on train and test set: ', difference)

# use Ridge() to perform ridge regression
# 'alpha' assigns the regularization strength to the model
# 'max_iter' assigns maximum number of iterations for the model to run
ridge = Ridge(alpha=1, max_iter=500)

# fit the model on train set
ridge.fit(X_train, y_train)

# print RMSE for test set
# call the function 'get_test_rmse'
print('RMSE on test set:', get_test_rmse(ridge))

# use Lasso() to perform lasso regression
# 'alpha' assigns the regularization strength to the model
# 'max_iter' assigns maximum number of iterations for the model to run
lasso = Lasso(alpha=0.01, max_iter=500)

# fit the model on train set
lasso.fit(X_train, y_train)

# print RMSE for test set
# call the function 'get_test_rmse'
print('RMSE on test set:', get_test_rmse(lasso))

# import function for elastic net regression
from sklearn.linear_model import ElasticNet

# use ElasticNet() to perform Elastic Net regression
# 'alpha' assigns the regularization strength to the model
# 'l1_ratio' is the ElasticNet mixing parameter
# 'l1_ratio = 0' performs Ridge regression
# 'l1_ratio = 1' performs Lasso regression
# pass number of iterations to 'max_iter'
enet = ElasticNet(alpha=0.1, l1_ratio=0.01, max_iter=500)

# fit the model on train data
enet.fit(X_train, y_train)

# print RMSE for test set
# call the function 'get_test_rmse'
print('RMSE on test set:', get_test_rmse(enet))

# create a dictionary with hyperparameters and its values
# 'alpha' assigns the regularization strength to the model
# 'max_iter' assigns maximum number of iterations for the model to run
tuned_paramaters = [{'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.1, 1, 5, 19, 40, 60, 80, 100]}]

# initiate the ridge regression model
ridge = Ridge()

# use GridSearchCV() to find the optimal value of alpha
# estimator: pass the ridge regression model
# param_grid: pass the list 'tuned_parameters'
# cv: number of folds in k-fold i.e. here cv = 10
ridge_grid = GridSearchCV(estimator=ridge,
                          param_grid=tuned_paramaters,
                          cv=5)

# fit the model on X_train and y_train using fit()
ridge_grid.fit(X_train, y_train)

# get the best parameters
print('Best parameters for Ridge Regression: ', ridge_grid.best_params_, '\n')

# print the RMSE for test set using the model having optimal value of alpha
print('RMSE on test set:', get_test_rmse(ridge_grid))

# create a dictionary with hyperparameters and its values
# 'alpha' assigns the regularization strength to the model
# 'max_iter' assigns maximum number of iterations for the model to run
tuned_paramaters = [{'alpha': [1e-15, 1e-10, 1e-8, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 20]}]

# 'max_iter':100,500,1000,1500,2000

# initiate the lasso regression model
lasso = Lasso()

# use GridSearchCV() to find the optimal value of alpha
# estimator: pass the lasso regression model
# param_grid: pass the list 'tuned_parameters'
# cv: number of folds in k-fold i.e. here cv = 10
lasso_grid = GridSearchCV(estimator=lasso,
                          param_grid=tuned_paramaters,
                          cv=10)

# fit the model on X_train and y_train using fit()
lasso_grid.fit(X_train, y_train)

# get the best parameters
print('Best parameters for Lasso Regression: ', lasso_grid.best_params_, '\n')

# print the RMSE for the test set using the model having optimal value of alpha
print('RMSE on test set:', get_test_rmse(lasso_grid))

# create a dictionary with hyperparameters and its values
# 'alpha' assigns the regularization strength to the model
# 'l1_ratio' is the ElasticNet mixing parameter
# 'max_iter' assigns maximum number of iterations for the model to run
tuned_paramaters = [{'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 20, 40, 60],
                     'l1_ratio': [0.0001, 0.0002, 0.001, 0.01, 0.1, 0.2]}]

# initiate the elastic net regression model
enet = ElasticNet()

# use GridSearchCV() to find the optimal value of alpha and l1_ratio
# estimator: pass the elastic net regression model
# param_grid: pass the list 'tuned_parameters'
# cv: number of folds in k-fold i.e. here cv = 10
enet_grid = GridSearchCV(estimator=enet,
                         param_grid=tuned_paramaters,
                         cv=10)

# fit the model on X_train and y_train using fit()
enet_grid.fit(X_train, y_train)

# get the best parameters
print('Best parameters for Elastic Net Regression: ', enet_grid.best_params_, '\n')

# print the RMSE for the test set using the model having optimal value of alpha and l1-ratio
print('RMSE on test set:', get_test_rmse(enet_grid))

rfr = RandomForestRegressor()
reg_model = rfr.fit(X_train, y_train)

# print the R-squared value for the model
# score() returns the R-squared value
reg_model.score(X_train, y_train)

# print training RMSE
print('RMSE on train set: ', get_train_rmse(reg_model))

# print training RMSE
print('RMSE on test set: ', get_test_rmse(reg_model))

# calculate the difference between train and test set RMSE
difference = abs(get_test_rmse(reg_model) - get_train_rmse(reg_model))

# print the difference between train and test set RMSE
print('Difference between RMSE on train and test set: ', difference)

# import various functions from sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# import the XGBoost function for classification
from xgboost import XGBClassifier

!pip
install
xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Instantiate the XGBRegressor
xgb_model = XGBRegressor(
    max_depth=5,  # Maximum depth of the trees
    min_child_weight=1,  # Minimum sum of instance weight needed in a child
    gamma=0.1,  # Minimum loss reduction required to make a further partition on a leaf node
    subsample=0.8,  # Subsample ratio of the training instances
    colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
    reg_alpha=0,  # L1 regularization term on weights
    reg_lambda=1,  # L2 regularization term on weights
    learning_rate=0.1,  # Learning rate (eta) for boosting
    n_estimators=100  # Number of trees to fit
)

# Fit the model on the training data
xgb_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = xgb_model.predict(X_test)

# Calculate mean squared error (MSE) as a metric
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# print training RMSE
print('RMSE on train set: ', get_train_rmse(xgb_model))

# print training RMSE
print('RMSE on test set: ', get_test_rmse(xgb_model))

# calculate the difference between train and test set RMSE
difference = abs(get_test_rmse(reg_model) - get_train_rmse(xgb_model))

# print the difference between train and test set RMSE
print('Difference between RMSE on train and test set: ', difference)

# print the R-squared value for the model
# score() returns the R-squared value
xgb_model.score(X_train, y_train)

# Instantiate the Random Forest Regressor with desired parameters
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2,
                                 random_state=42)

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_model.predict(X_test)

# Evaluate the model's performance
# print the R-squared value for the model
# score() returns the R-squared value
rf_model.score(X_train, y_train)

# print training RMSE
print('RMSE on train set: ', get_train_rmse(rf_model))

# print training RMSE
print('RMSE on test set: ', get_test_rmse(rf_model))

# calculate the difference between train and test set RMSE
difference = abs(get_test_rmse(rf_model) - get_train_rmse(rf_model))

# print the difference between train and test set RMSE
print('Difference between RMSE on train and test set: ', difference)

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [5, 10, 15],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Instantiate the Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Instantiate GridSearchCV with the Random Forest Regressor and parameter grid
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Instantiate a new Random Forest Regressor with the best hyperparameters
best_rf_model = RandomForestRegressor(**best_params, random_state=42)

# Fit the best model on the training data
best_rf_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = best_rf_model.predict(X_test)

# Evaluate the model's performance
# print training RMSE
print('RMSE on train set: ', get_train_rmse(best_rf_model))

# print training RMSE
print('RMSE on test set: ', get_test_rmse(best_rf_model))

# calculate the difference between train and test set RMSE
difference = abs(get_test_rmse(best_rf_model) - get_train_rmse(best_rf_model))

# print the difference between train and test set RMSE
print('Difference between RMSE on train and test set: ', difference)

best_rf_model.score(X_train, y_train)

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [5, 10, 15],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Instantiate the Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Instantiate GridSearchCV with the Random Forest Regressor and parameter grid
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=10, scoring='neg_root_mean_squared_error',
                           n_jobs=-1)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Instantiate a new Random Forest Regressor with the best hyperparameters
best_rff_model = RandomForestRegressor(**best_params, random_state=42)

# Fit the best model on the training data
best_rff_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = best_rff_model.predict(X_test)

# Evaluate the model's performance
# print training RMSE
print('RMSE on train set: ', get_train_rmse(best_rff_model))

# print training RMSE
print('RMSE on test set: ', get_test_rmse(best_rff_model))

# calculate the difference between train and test set RMSE
difference = abs(get_test_rmse(best_rff_model) - get_train_rmse(best_rff_model))

# print the difference between train and test set RMSE
print('Difference between RMSE on train and test set: ', difference)

best_rff_model.score(X_train, y_train)

print('best parameter for random forest regressior:', grid_search.best_params_, '\n')

rfr_model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=2, min_samples_leaf=2,
                                  random_state=42)
rfr_model.fit(X_train, y_train)
y_pred = rfr_model.predict(X_test)

# Evaluate the model's performance
# print training RMSE
print('RMSE on train set: ', get_train_rmse(rfr_model))

# print training RMSE
print('RMSE on test set: ', get_test_rmse(rfr_model))

# calculate the difference between train and test set RMSE
difference = abs(get_test_rmse(rfr_model) - get_train_rmse(rfr_model))

# calculate the accuracy
print('accuracy for this rfr_model:', rfr_model.score(X_train, y_train))

