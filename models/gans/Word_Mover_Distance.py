import os
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk import download

download('stopwords')
stop_words = stopwords.words('english')

def word_movers_distance(model, text1, text2):
	text1 = [w for w in text1 if w not in stop_words]
	text2 = [w for w in text2 if w not in stop_words]
	# model.init_sims(replace=True)

	distance = model.wmdistance(text1, text2)

	return distance