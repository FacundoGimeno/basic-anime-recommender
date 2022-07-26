import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('anime.csv')

# Convert into a single string
def get_important_features(data):
    important_features = []
    for i in range(0, data.shape[0]):
        important_features.append(data['Genres'][i] + ' ' + data['Name'][i] + ' ' + str(data['ID'][i]) + ' ' + str(data['Episodes'][i]) + ' ' + str(data['Score'][i]))
    return important_features

df['important_features'] = get_important_features(df)

# Convert text into a matrix of word counts
cm = CountVectorizer().fit_transform(df['important_features'])

# Get the cosine similarity matrix
cs = cosine_similarity(cm)

# Anime that user like
user = input('Enter an anime that you like: ')

# Find anime id of user's input
user_id = df[df.Name == user]['ID'].values[0]

# Create a list of scores
scores = list(enumerate(cs[user_id]))

# Sort the list by the scores
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
sorted_scores = sorted_scores[1:]

# Print the first 3 anime recommendations
print('Top 3 recommendations:')
j = 0
for i in sorted_scores:
    anime_title = df[df['ID'] == i[0]]['Name'].values[0]
    print(anime_title)
    if j > 1:
        break
    j += 1