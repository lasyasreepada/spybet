import pandas as pd
import numpy as np
# ! pip3 install sportsreference
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge

""" DATA PREPARATION """
# Load 2021 data.
df_season = pd.read_csv('nhl-2021-season.csv')
df_playoffs = pd.read_csv('nhl-2021-playoffs.csv')

df = df_playoffs

# Check Data
df.head()

# cast date column into standard datetime format.
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

df['goal_diff'] = df['G.1'] - df['G']

# create new variables to show home team win or loss result
df['home_win'] = np.where(df['goal_diff'] > 0, 1, 0)
df['home_loss'] = np.where(df['goal_diff'] < 0, 1, 0)

df_visitor = pd.get_dummies(df['Visitor'], dtype=np.int64)
df_home = pd.get_dummies(df['Home'], dtype=np.int64)

# subtract home from visitor
df_model = df_home.sub(df_visitor)
df_model['goal_diff'] = df['goal_diff']

""" PREDICTIVE MODEL """

# Simple ridge regression boooo
df_train = df_model.copy() # just to make sure I don't fuck something up and have to reload data

lr = Ridge(alpha=0.01)
X = df_train.dropna(subset=['goal_diff']).drop(['goal_diff'], axis=1)
y = df_train['goal_diff'].dropna()

np.shape(X)
np.shape(y)

lr.fit(X, y)

df_ratings = pd.DataFrame(data={'team': X.columns, 'rating': lr.coef_})
df_ratings.sort_values(by=['rating'], ascending='False')

# TODO
# Consider or weight recent games greater
# Consider team schedule travel vs home
# Add cooler models like NN, GBM, RF, SVR -- which models would be most appropriate for the research question?
# Allow for user input aka gut feeling
