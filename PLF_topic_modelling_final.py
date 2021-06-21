#####
# Topic Modelling for PLF Research Field - Investigating the inclusion of ethical and social terms into peer-reviewed publications
# Tested on: Windows 10 Insider Preview Build 21390, Anaconda Python 3.8, Ryzen 5950X (16-cores), 128GB DDR4 Ram, RTX2070Super
#####


# Import  necessary libraries
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import spacy
import nltk
import re
import string
import pandas as pd
import numpy as np
import gensim
from gensim import corpora
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from pprint import pprint
from nltk.corpus import stopwords

# Load libraries for "language cleaning"
nlp = spacy.load("en")
##Stop words
stop_words = stopwords.words("english")
##Add additional terms to the list of words to be excluded (could, and should be extended after the manual examination of initial text data /
# to remove highly specific terms not contributing to the topic interpretability)
stop_words.extend(
    [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "about",
        "across",
        "after",
        "all",
        "also",
        "an",
        "and",
        "another",
        "added",
        "any",
        "are",
        "as",
        "at",
        "basically",
        "be",
        "because",
        "become",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "by",
        "came",
        "can",
        "come",
        "could",
        "did",
        "do",
        "does",
        "each",
        "else",
        "every",
        "either",
        "especially",
        "for",
        "from",
        "get",
        "given",
        "gets",
        "give",
        "gives",
        "got",
        "goes",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "him",
        "himself",
        "his",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "lands",
        "like",
        "make",
        "making",
        "made",
        "many",
        "may",
        "me",
        "might",
        "more",
        "most",
        "much",
        "must",
        "my",
        "never",
        "provide",
        "provides",
        "perhaps",
        "no",
        "now",
        "of",
        "on",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "re",
        "said",
        "same",
        "see",
        "should",
        "since",
        "so",
        "some",
        "still",
        "such",
        "seeing",
        "see",
        "take",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "up",
        "use",
        "using",
        "used",
        "underway",
        "very",
        "want",
        "was",
        "way",
        "we",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "whilst",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
        "etc",
        "via",
        "eg",
        "rr",
        "brpm",
        "datum",
        "etm",
        "avhrr",
        "iconos",
        "modis",
        "pc",
        "kg",
        "ha",
        "per",
        "ph",
        "icp",
        "thus",
        "na",
        "ece",
        "km",
        "utm",
        "hr",
        "ml",
        "bmp",
        "irtv",
        "ii",
        "cielab",
        "among",
        "ever",
        "paper",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "article",
        "editor",
        "author",
        "review",
        "MDPI",
        "2019",
        "conference",
        "elsevi",
        "elsevier",
        "th",
        "european",
        "mdpi",
        "basel",
        "licensee",
        "basel",
        "switzerland",
        "ecplf",
        "springer",
        "taylor",
        "francis",
        "academic",
        "publisher",
        "editorial",
        "department",
        "creative",
        "commons",
        "wageningen",
        "attribution",
        "asabe",
        "annual",
        "international",
        "meeting",
        "cambridge",
        "university",
        "press",
        "license",
        "permit",
        "unrestricted",
        "globe",
        "cooperative",
        "research",
        "india",
        "rme",
        "tdb",
        "dft",
        "review",
        "weibern",
        "austria",
        "italy",
    ]
)

# Read the data
df = pd.read_excel("analysis.xlsx")
##Drop everything but target column with Abstracts
df.drop(
    columns=["Authors", "Title", "Year", "Author Keywords", "Index Keywords"],
    inplace=True,
)
##Examine the dataframe
display(df.head(10))
df.dtypes

# Clean and prepare data - main function
def cleaning(df, col_name):
    # lowercase text data in target column
    df[col_name] = df[col_name].map(lambda x: x.lower())
    # lemmatize words
    df[col_name] = df[col_name].astype(str).map(lemmatize)
    # remove punctuation
    df[col_name] = df[col_name].map(punctuation)
    return df

##Sub-function for cleaning
def punctuation(comment):
    regex = re.compile(
        "[" + re.escape("!\"#%&'()*+,-./:;<=>?@©[\\]^_`{|}~") + "0-9\\r\\t\\n]"
    )
    nopunct = regex.sub(" ", comment)
    nopunct_words = nopunct.split(" ")
    filter_words = [word.strip() for word in nopunct_words if word != ""]
    words = " ".join(filter_words)
    return words

####Sub-function for cleaning
def lemmatize(comment):
    lemmatized = nlp(comment)
    lemmatized_final = " ".join(
        [word.lemma_ for word in lemmatized if word.lemma_ != "'s"]
    )
    return lemmatized_final

# Process dataframe and prepare it for LDA analysis
abstracts = pd.DataFrame(df.Abstract)
clean_abstracts = cleaning(abstracts, "Abstract")
clean_abstracts.head()

# Important to examine the produced bigrams and trigrams to see that the terms from stop_words list are removed /
# and that the examples make sence, since it will affect the interpretability of the final LDA model

# Make bigrams
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_documents(
    [comment.split() for comment in clean_abstracts.Abstract]
)
##Filter only those that occur at least 5 times
finder.apply_freq_filter(5)
bigram_scores = finder.score_ngrams(bigram_measures.pmi)
bigram_pmi = pd.DataFrame(bigram_scores)
bigram_pmi.columns = ["bigram", "pmi"]
bigram_pmi.sort_values(by="pmi", axis=0, ascending=False, inplace=True)
##Filter for bigrams with only noun-type structures
def bigram_filter(bigram):
    tag = nltk.pos_tag(bigram)
    if tag[0][1] not in ["JJ", "NN"] and tag[1][1] not in ["NN"]:
        return False
    if bigram[0] in stop_words or bigram[1] in stop_words:
        return False
    if "n" in bigram or "t" in bigram:
        return False
    if "PRON" in bigram:
        return False
    return True

# Make trigrams
trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = nltk.collocations.TrigramCollocationFinder.from_documents(
    [comment.split() for comment in clean_abstracts.Abstract]
)
##Filter only those that occur at least 5 times
finder.apply_freq_filter(5)
trigram_scores = finder.score_ngrams(trigram_measures.pmi)
trigram_pmi = pd.DataFrame(trigram_scores)
trigram_pmi.columns = ["trigram", "pmi"]
trigram_pmi.sort_values(by="pmi", axis=0, ascending=False, inplace=True)
##Filter for trigrams with only noun-type structures
def trigram_filter(trigram):
    tag = nltk.pos_tag(trigram)
    if tag[0][1] not in ["JJ", "NN"] and tag[1][1] not in ["JJ", "NN"]:
        return False
    if (
        trigram[0] in stop_words
        or trigram[-1] in stop_words
        or trigram[1] in stop_words
    ):
        return False
    if "n" in trigram or "t" in trigram:
        return False
    if "PRON" in trigram:
        return False
    return True

# Choose top 500* ngrams in this case ranked by PMI that have noun-type structures
nltk.download("averaged_perceptron_tagger")

filtered_bigram = bigram_pmi[
    bigram_pmi.apply(
        lambda bigram: bigram_filter(bigram["bigram"]) and bigram.pmi > 5, axis=1
    )
][:500]

filtered_trigram = trigram_pmi[
    trigram_pmi.apply(
        lambda trigram: trigram_filter(trigram["trigram"]) and trigram.pmi > 5, axis=1
    )
][:500]

bigrams = [
    " ".join(x) for x in filtered_bigram.bigram.values if len(x[0]) > 2 or len(x[1]) > 2
]
trigrams = [
    " ".join(x)
    for x in filtered_trigram.trigram.values
    if len(x[0]) > 2 or len(x[1]) > 2 and len(x[2]) > 2
]

##Examples of bigrams
bigrams[:10]
##Examples of trigrams
trigrams[:10]

##Concatenate n-grams
def replace_ngram(x):
    for gram in trigrams:
        x = x.replace(gram, "_".join(gram.split()))
    for gram in bigrams:
        x = x.replace(gram, "_".join(gram.split()))
    return x


abstracts_w_ngrams = clean_abstracts.copy()
abstracts_w_ngrams.Abstract = abstracts_w_ngrams.Abstract.map(
    lambda x: replace_ngram(x)
)

# tokenize Abstractss + remove stop words + remove names + remove words with less than 2 characters
abstracts_w_ngrams = abstracts_w_ngrams.Abstract.map(
    lambda x: [word for word in x.split() if word not in stop_words and len(word) > 2]
)

abstracts_w_ngrams.head()

# Filter for only nouns
def noun_only(x):
    pos_comment = nltk.pos_tag(x)
    filtered = [word[0] for word in pos_comment if word[1] in ["NN"]]
    # to filter both noun and verbs
    # filtered = [word[0] for word in pos_comment if word[1] in ['NN','VB', 'VBD', 'VBG', 'VBN', 'VBZ']]
    return filtered

# Final dataframe for LDA modelling
final_abstracts = abstracts_w_ngrams.map(noun_only)

# LDA Model
dictionary = corpora.Dictionary(final_abstracts)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in final_abstracts]

##Evaluate number of topics for the best coherence - 50 passes
coherence = []
for k in range(1, 25):
    print("Round: " + str(k))
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(
        doc_term_matrix,
        num_topics=k,
        id2word=dictionary,
        passes=50,
        iterations=200,
        chunksize=100,
        eval_every=1,
    )

    cm = gensim.models.coherencemodel.CoherenceModel(
        model=ldamodel, texts=final_abstracts, dictionary=dictionary, coherence="c_v"
    )
    coherence.append((k, cm.get_coherence()))

##Plot the results for 50 passes
x_val = [x[0] for x in coherence]
y_val = [x[1] for x in coherence]
plt.plot(x_val, y_val)
plt.scatter(x_val, y_val)
plt.title("Number of Topics vs. Coherence")
plt.xlabel("Number of Topics")
plt.ylabel("Coherence")
plt.xticks(x_val)
plt.show()

##Evaluate number of topics for the best coherence - 250 passes
coherence = []
for k in range(1, 25):
    print("Round: " + str(k))
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(
        doc_term_matrix,
        num_topics=k,
        id2word=dictionary,
        passes=250,
        iterations=200,
        chunksize=100,
        eval_every=1,
    )

    cm = gensim.models.coherencemodel.CoherenceModel(
        model=ldamodel, texts=final_abstracts, dictionary=dictionary, coherence="c_v"
    )
    coherence.append((k, cm.get_coherence()))

##Plot the results for 250 passes
x_val = [x[0] for x in coherence]
y_val = [x[1] for x in coherence]
plt.plot(x_val, y_val)
plt.scatter(x_val, y_val)
plt.title("Number of Topics vs. Coherence")
plt.xlabel("Number of Topics")
plt.ylabel("Coherence")
plt.xticks(x_val)
plt.show()

# LDA Model implementation
##LDA Model 1 - 13 Topics
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(
    doc_term_matrix,
    num_topics=13,
    id2word=dictionary,
    passes=50,
    iterations=200,
    chunksize=100,
    eval_every=1,
    random_state=1,
)
##LDA Model 2 - 3 Topics
Lda2 = gensim.models.ldamodel.LdaModel
ldamodel2 = Lda2(
    doc_term_matrix,
    num_topics=3,
    id2word=dictionary,
    passes=250,
    iterations=200,
    chunksize=100,
    eval_every=1,
    random_state=1,
)

##Show the results with 10 most prominent terms
ldamodel.show_topics(13, num_words=10, formatted=False)
ldamodel2.show_topics(3, num_words=10, formatted=False)

# LDA Model visualisation with pyLDAvis
LDA_Model1 = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, mds="pcoa")
pyLDAvis.display(LDA_Model1)
pyLDAvis.save_html(LDA_Model1, "LDA_topics_13.html")

LDA_Model2 = pyLDAvis.gensim.prepare(ldamodel2, doc_term_matrix, dictionary, mds="pcoa")
pyLDAvis.display(LDA_Model2)
pyLDAvis.save_html(LDA_Model2, "LDA_topics_3.html")

##Simple LDA Model 1 representation
all_topics = {}
num_terms = 5  # Adjust number of words to represent each topic
lambd = 0.6  # Adjust this accordingly based on tuning above
for i in range(
    1, 13
):  # Adjust this to reflect number of topics chosen for final LDA model
    topic = LDA_Model1.topic_info[
        LDA_Model1.topic_info.Category == "Topic" + str(i)
    ].copy()
    topic["relevance"] = topic["loglift"] * (1 - lambd) + topic["logprob"] * lambd
    all_topics["Topic " + str(i)] = (
        topic.sort_values(by="relevance", ascending=False).Term[:num_terms].values
    )
pd.DataFrame(all_topics).T

##Simple LDA Model 2 representation
all_topics = {}
num_terms = 5  # Adjust number of words to represent each topic
lambd = 0.6  # Adjust this accordingly based on tuning above
for i in range(
    1, 13
):  # Adjust this to reflect number of topics chosen for final LDA model
    topic = LDA_Model2.topic_info[
        LDA_Model2.topic_info.Category == "Topic" + str(i)
    ].copy()
    topic["relevance"] = topic["loglift"] * (1 - lambd) + topic["logprob"] * lambd
    all_topics["Topic " + str(i)] = (
        topic.sort_values(by="relevance", ascending=False).Term[:num_terms].values
    )
pd.DataFrame(all_topics).T
