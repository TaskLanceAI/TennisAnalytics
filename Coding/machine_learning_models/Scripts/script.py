import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

# load and investigate the data here:
tennis = pd.read_csv('tennis_stats.csv')
print(tennis.head(5))
print(tennis['Player'].nunique())
print(len(tennis))

# perform exploratory analysis here:
features = ['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon','SecondServePointsWon','SecondServeReturnPointsWon','Aces','BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities','BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon','ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon','TotalServicePointsWon','Wins','Losses','Ranking']
row_count = 0
for i in features:
  #row_count += 1
  #plt.subplot(7,3,row_count)
  sns.scatterplot(x=i, y="Winnings", data=tennis)
  plt.show()
  plt.clf()

## perform single feature linear regressions here:
features = tennis[['ServiceGamesPlayed']]
outcome = tennis[['Winnings']]
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)
model = LinearRegression()
model.fit(features_train,outcome_train)
model.score(features_test,outcome_test)
print('ServiceGamesPlayed vs winnings score: ',round(model.score(features_test,outcome_test),2))

prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
# Show the plot
plt.show()

features = tennis[['ReturnGamesPlayed']]
outcome = tennis[['Winnings']]
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)
model = LinearRegression()
model.fit(features_train,outcome_train)
model.score(features_test,outcome_test)
print('ReturnGamesPlayed vs winnings score: ',round(model.score(features_test,outcome_test),2))

features = tennis[['DoubleFaults']]
outcome = tennis[['Winnings']]
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)
model = LinearRegression()
model.fit(features_train,outcome_train)
model.score(features_test,outcome_test)
print('DoubleFaults vs winnings score: ',round(model.score(features_test,outcome_test),2))

features = tennis[['BreakPointsOpportunities']]
outcome = tennis[['Winnings']]
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)
model = LinearRegression()
model.fit(features_train,outcome_train)
model.score(features_test,outcome_test)
print('BreakPointsOpportunities vs winnings score: ',round(model.score(features_test,outcome_test),2))

## perform two feature linear regressions here:
features = tennis[['ServiceGamesPlayed','ReturnGamesPlayed']]
outcome = tennis[['Winnings']]
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)
model = LinearRegression()
model.fit(features_train,outcome_train)
model.score(features_test,outcome_test)
print('ServiceGamesPlayed & ReturnGamesPlayed vs winnings score: ',round(model.score(features_test,outcome_test),2))

features = tennis[['DoubleFaults','BreakPointsOpportunities']]
outcome = tennis[['Winnings']]
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)
model = LinearRegression()
model.fit(features_train,outcome_train)
model.score(features_test,outcome_test)
print('DoubleFaults & BreakPointsOpportunities  vs winnings score: ',round(model.score(features_test,outcome_test),2))

## perform multiple feature linear regressions here:

features = tennis[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']]
outcome = tennis[['Winnings']]
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)
model = LinearRegression()
model.fit(features_train,outcome_train)
model.score(features_test,outcome_test)
print('All features  vs winnings score: ',round(model.score(features_test,outcome_test),2))