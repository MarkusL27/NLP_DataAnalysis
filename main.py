import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')


def preprocess_data(data):
    # alles lowercase
    data = data.str.lower()
    # Leerzeichen am Anfang entfernen
    data = data.str.strip()
    # Zahlen entfernen
    data = data.replace(r'\d+', '', regex=True)
    # Stoppwörter entfernen
    data = data.apply(remove_stopwords)
    # Lemmatisierung durchführen
    data = data.apply(do_lemmatizing)
    return data


def remove_stopwords(data):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(data.lower())
    return " ".join([word for word in words if word.isalpha() and word not in stop_words])


def do_lemmatizing(data):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(data)
    return " ".join([lemmatizer.lemmatize(word) for word in words if word.isalpha()])


def compare_vectors(bow, tfidf, vocabulary):
    top_themes = 10
    # BOW
    sum_themes_bow = np.array(bow.sum(axis=0)).flatten()
    top_words = vocabulary[sum_themes_bow.argsort()[::-1]][:top_themes]
    print("Meistverwendete Themen gemäß BoW:\n", top_words)
    # TF-IDF
    sum_themes_tfidf = np.array(tfidf.sum(axis=0)).flatten()
    top_words = vocabulary[sum_themes_tfidf.argsort()[::-1]][:top_themes]
    print("Meistverwendete Themen laut TF-IDF:\n", top_words)


def create_lsa(data):
    # TF-IDF Vektorisierung
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vector = tfidf_vectorizer.fit_transform(data)
    # LSA mit TruncatedSVD
    lsa_model = TruncatedSVD(n_components=5, random_state=42)
    reduced = lsa_model.fit_transform(tfidf_vector)
    terms = tfidf_vectorizer.get_feature_names_out()

    print("Ergebnis Themenmodellierung LSA:")
    for topic_idx, topic in enumerate(lsa_model.components_):
        top_words = [terms[i] for i in topic.argsort()[:-5 - 1:-1]]
        print(f"Thema {topic_idx + 1}: {', '.join(top_words)}")


def create_lda(data):
    # Sätze aufsplitten
    words = data.apply(word_tokenize)
    dictionary = gensim.corpora.Dictionary(words)
    bow_corpus = [dictionary.doc2bow(doc) for doc in words]
    lda_model = gensim.models.LdaModel(corpus=bow_corpus, num_topics=5, id2word=dictionary)
    topics = lda_model.print_topics()
    print("Ergebnis Themenmodellierung LDA:")
    for topic in topics:
        print(topic)


def load_data():
    # CSV als Pandas Dataframe einlesen
    df = pd.read_csv('data/Datasetprojpowerbi.csv')
    # Nur Spalte Reports interessant
    data = df['Reports']
    return data


def main():
    data = load_data()
    preprocessed_data = preprocess_data(data)

    # Bag of words Vektor erstellen
    vectorizer = CountVectorizer()
    bag_of_words = vectorizer.fit_transform(preprocessed_data)

    # TF-IDF Vektor erstellen
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vector = tfidf_vectorizer.fit_transform(preprocessed_data)

    vocabulary = vectorizer.get_feature_names_out()

    # die beiden Vektoren u. Vokabular übergeben
    compare_vectors(bag_of_words, tfidf_vector, vocabulary)

    create_lda(preprocessed_data)
    create_lsa(preprocessed_data)


if __name__ ==  '__main__':
    main()