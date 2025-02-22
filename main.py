import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models import CoherenceModel

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)



def load_data():
    """
    Lädt die CSV Datei als Pandas Dataframe
    Nimmt nur die relevante Spalte 'Reports'
    :return: Pandas Series
    """
    df = pd.read_csv('data/Datasetprojpowerbi.csv')
    data = df['Reports']
    return data


def preprocess_data(data):
    """
    Textvorverarbeitung der Daten
    Kleinschreibung, Leerzeichen am Anfang entfernen, Zahlen sowie Stopwörter entfernen und Lemmatisierung durchführen
    :param data: Pandas Series
    :return: Pandas Series
    """
    data = data.str.lower()
    data = data.str.strip()
    data = data.replace(r'\d+', '', regex=True)
    data = data.apply(remove_stopwords, language='english')
    data = data.apply(do_lemmatization)
    return data


def remove_stopwords(data, language):
    """
    Entfernt die Stopwörter der angegebenen Sprache
    :param data: Pandas Series
    :param language: String
    :return: Pandas Series
    """
    stop_words = set(stopwords.words(language))
    words = word_tokenize(data.lower())
    return " ".join([word for word in words if word.isalpha() and word not in stop_words])


def do_lemmatization(data):
    """
    Lemmatisierung mithilfe des nltk WordNetLemmatizer
    :param data: Pandas Series
    :return: Pandas Series
    """
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(data)
    return " ".join([lemmatizer.lemmatize(word) for word in words if word.isalpha()])


def compare_vectors(bow, tfidf, vocabulary):
    """
    Bildet eine vertikale Summierung der Werte der beiden Vektoren
    Gibt jeweils die so gebildeten Top 10 Begriffe der Vektoren aus
    :param bow: Scipy csr matrix
    :param tfidf: Scipy csr matrix
    :param vocabulary: Numpy ndarray
    :return: None
    """
    top_themes = 10
    # BOW
    sum_themes_bow = np.array(bow.sum(axis=0)).flatten()
    top_words = vocabulary[sum_themes_bow.argsort()[::-1]][:top_themes]
    print("Meistverwendete Themen laut BoW:\n", top_words)
    # TF-IDF
    sum_themes_tfidf = np.array(tfidf.sum(axis=0)).flatten()
    top_words = vocabulary[sum_themes_tfidf.argsort()[::-1]][:top_themes]
    print("Meistverwendete Themen laut TF-IDF:\n", top_words)


def get_optimum_num_topics_lda(data, topic_range):
    """
    Ermittelt die optimale Anzahl an Themen für ein LDA Model für einen gegebenen Datensatz in einer bestimmten Range an Themenzahl
    :param data: Pandas Series
    :param topic_range: Range
    :return: int
    """
    coherence_values = []
    words = data.apply(word_tokenize)
    dictionary = gensim.corpora.Dictionary(words)
    for num_topics in topic_range:
        lda_model = get_lda_model(data, num_topics=num_topics)
        coherence_model = CoherenceModel(model=lda_model, texts=words, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
    optimal_k = topic_range[coherence_values.index(max(coherence_values))]
    return optimal_k


def get_lda_model(data, num_topics):
    """
    Liefert ein Gensim LdaModel für einen Datensatz mit einer bestimmten Anzahl an Themen zurück
    :param data: Pandas Series
    :param num_topics: int
    :return: gensim LdaModel
    """
    words = data.apply(word_tokenize)
    dictionary = gensim.corpora.Dictionary(words)
    bow_corpus = [dictionary.doc2bow(doc) for doc in words]
    lda_model = gensim.models.LdaModel(corpus=bow_corpus, num_topics=num_topics, id2word=dictionary, passes=20)
    return lda_model


def get_optimum_num_topics_lsa(data, topic_range):
    """
    Ermittelt die optimale Anzahl an Themen für ein LSA Model für einen gegebenen Datensatz in einer bestimmten Range an Themenzahl
    :param data: Pandas Series
    :param topic_range: Range
    :return: int
    """
    # TF-IDF Vektorisierung
    vectorizer = TfidfVectorizer()
    tfidf_vector = vectorizer.fit_transform(data)
    feature_names = np.array(vectorizer.get_feature_names_out())

    tokenized_docs = [doc.split() for doc in data]

    coherence_scores = []

    for n_topics in topic_range:
        svd = TruncatedSVD(n_components=n_topics, random_state=42)
        svd.fit(tfidf_vector)

        # Top 10 Wörter extrahieren
        topics = []
        for topic in svd.components_:
            top_words = feature_names[np.argsort(topic)][-10:]
            topics.append(top_words.tolist())

        # Kohärenz berechnen
        coherence_scores.append(get_coherence_svd(topics, tokenized_docs))

    optimal_num = topic_range[coherence_scores.index(max(coherence_scores))]
    return optimal_num


def get_coherence_svd(topics, tokenized_docs):
    """
    Berechnet mithilfe des gensim CoherenceModel den Kohärenzwert für die übergebenen Themen
    :param topics: list
    :param tokenized_docs: list
    :return: float
    """
    dictionary = gensim.corpora.Dictionary(tokenized_docs)
    coherence_model = CoherenceModel(topics=topics, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    return coherence_score


def extract_topics_lda(data, num_topics):
    """
    Extrahiert die Anzahl an gewünschten Themen eines Datensatzes mithilfe des gensim LdaModels
    :param data: Pandas Series
    :param num_topics: int
    :return: None
    """
    words = data.apply(word_tokenize)
    dictionary = gensim.corpora.Dictionary(words)
    bow_corpus = [dictionary.doc2bow(doc) for doc in words]
    lda_model = gensim.models.LdaModel(corpus=bow_corpus, num_topics=num_topics, id2word=dictionary, passes=50)
    topics = lda_model.print_topics()
    print("Ergebnis Themenmodellierung LDA:")
    for topic in topics:
        print(topic)


def extract_topics_lsa(data, num_topics):
    """
    Extrahiert die Anzahl an gewünschten Themen eines Datensatzes mithilfe des TruncatedSVD
    :param data: Pandas Series
    :param num_topics: int
    :return: None
    """
    # TF-IDF Vektorisierung
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vector = tfidf_vectorizer.fit_transform(data)
    # LSA mit TruncatedSVD
    lsa_model = TruncatedSVD(n_components=num_topics, random_state=42)
    reduced = lsa_model.fit_transform(tfidf_vector)
    terms = tfidf_vectorizer.get_feature_names_out()

    print("Ergebnis Themenmodellierung LSA:")
    for topic_idx, topic in enumerate(lsa_model.components_):
        top_words = [terms[i] for i in topic.argsort()[:-10 - 1:-1]]
        print(f"Thema {topic_idx + 1}: {', '.join(top_words)}")


def main():
    data = load_data()
    preprocessed_data = preprocess_data(data)

    # Bag of words Vektor erstellen
    vectorizer = CountVectorizer()
    bag_of_words = vectorizer.fit_transform(preprocessed_data)

    # TF-IDF Vektor erstellen
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vector = tfidf_vectorizer.fit_transform(preprocessed_data)

    # die beiden Vektoren u. Vokabular übergeben
    vocabulary = vectorizer.get_feature_names_out()
    compare_vectors(bag_of_words, tfidf_vector, vocabulary)

    # die optimale Anzahl an Themen finden für LDA
    #optimum_numbers_lda = get_optimum_num_topics_lda(preprocessed_data, range(2,20))
    #print(optimum_numbers_lda)

    # die optimale Anzahl an Themen finden für LSA
    #optimum_numbers_lsa = get_optimum_num_topics_lsa(preprocessed_data, range(2,20))
    #print(optimum_numbers_lsa)

    # Themenextraktion LDA
    extract_topics_lda(preprocessed_data, 10)
    # Themenextraktion LSA
    extract_topics_lsa(preprocessed_data, 3)


if __name__ ==  '__main__':
    main()