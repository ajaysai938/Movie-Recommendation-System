import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
data = {
    'title': ['Inception', 'The Matrix', 'Interstellar', 'The Prestige', 'The Dark Knight'],
    'description': [
        'A thief who steals corporate secrets through dream-sharing technology.',
        'A computer hacker learns about the true nature of his reality.',
        'A team travels through a wormhole in space to ensure humanity’s survival.',
        'Two magicians engage in a battle to create the ultimate illusion.',
        'Batman raises the stakes in his war on crime.'
    ]
}
df = pd.DataFrame(data)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])
cosine_sim = cosine_similarity(tfidf_matrix)
def recommend(title):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]
print(recommend("Inception"))