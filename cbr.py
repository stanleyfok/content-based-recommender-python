# references:
# http://scikit-learn.org/stable/auto_examples/text/document_clustering.html
# http://blog.untrod.com/2016/06/simple-similar-products-recommendation-engine-in-python.html

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from time import time

df = pd.read_csv('data/job_titles.csv', encoding='ISO-8859-1', index_col=0)

# turn index to str
df.index = df.index.map(str)

t0 = time()

tf = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 3),
                     min_df=0,
                     stop_words='english')

tfidf_matrix = tf.fit_transform(df['title'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

data = {}
for idx, row in enumerate(df.iterrows()):
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [{'id': df.index[i], 'score': cosine_similarities[idx][i]}
                     for i in similar_indices]

    # exclude the first element as it is self
    similar_items = similar_items[1:]

    data[df.index[idx]] = similar_items

print("done in %fs" % (time() - t0))

# get similar items for the job with id = "1"
print(data["1"])
