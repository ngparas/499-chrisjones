"""Classes and Functions for EECS 499
"""
import json
import spacy
import re
import urllib.parse as ulp
import requests
from textblob import TextBlob
from nltk.corpus import stopwords
import pandas as pd
from textacy import text_stats
import textacy
from textacy import similarity
NLP = spacy.load('en')
stopwords = stopwords.words('english')

def clean_article_title(title):
    """Take an article-source URL and return a cleanly formatted title

    Args:
        title (string): a source URL
    Return:
        article_title (string): A cleanly formatted title to include in response
    """
    article_title = ulp.unquote(title)
    article_title = re.split('title=', article_title)[1].replace('+', ' ')
    title_len = len(article_title)
    if (article_title[title_len - 5:] in ['&amp', '&amp;']):
        article_title = article_title[:title_len - 5]
    return article_title

class Corpus(object):
    """A lightweight class that provides a consistent interface to copora

    Attributes:
        es_host (string): The URL and Port of the ES cluster
        author (string): The desired corpus, currently supports either 'jones' or 'ebert'

    """
    def __init__(self, author, es_host='http://localhost:9200/'):
        self.es_host = es_host

        if author in ['jones', 'ebert']:
            self.author = author
        else:
            raise ValueError('author must be one of "jones" or "ebert", you gave {}'.format(author))

    def retrieve_articles(self, keyword_string):
        """Retrieve articles with text relevant to the keyword string

        Args:
            keyword_string (string): A string with keywords, e.g. 'Tracy Letts' or 'Apocalypse Now'

        Return:
            A list of tuples: (title, article text)
        """

        if self.author == 'jones':
            q = {
                "_source": ["Full text:", "ProQ:"],
                "from": 0, "size": 10000,
                "query": {
                    "bool" : {
                        "must": [{"match":{"Full text:": i}} for i in keyword_string.split(' ')]
                    }
                }
            }
            r = json.loads(requests.post(self.es_host+"flattened-articles/_search", json.dumps(q)).content.decode('utf-8', 'ignore'))
            return [Article(clean_article_title(i['_source']['ProQ:']), i['_source']['Full text:']) for i in r['hits']['hits']]
        elif self.author == 'ebert':
            q = {
                "_source": ["text", "title"],
                "from": 0, "size": 10000,
                "query": {
                    "bool" : {
                        "must": [{"match":{"text": i}} for i in keyword_string.split(' ')]
                    }
                }
            }
            r = json.loads(requests.post(self.es_host+"ebert-reviews/_search", json.dumps(q)).content.decode('utf-8', 'ignore'))
            return [Article(i['_source']['title'], i['_source']['text']) for i in r['hits']['hits']]
        else:
            raise ValueError('author must be one of "jones" or "ebert", you gave {}'.format(author))

    def filter_articles_by_person(self, name):
        """Retrieve articles with a persons name

        Args:
            name (string): A string with a name e.g. 'Tracy Letts'

        Return:
            A list of tuples: (title, article text)
        """

        if self.author == 'jones':
            q = {
                "_source": ["Full text:", "ProQ:"],
                "from": 0, "size": 10000,
                "query": {
                    "bool" : {
                        "must": [{"match":{"Full text:": i}} for i in name.split(' ')]
                    }
                }
            }
            r = json.loads(requests.post(self.es_host+"flattened-articles/_search", json.dumps(q)).content.decode('utf-8', 'ignore'))
            return [Article(clean_article_title(i['_source']['ProQ:']), i['_source']['Full text:']) for i in r['hits']['hits']]
        elif self.author == 'ebert':
            q = {
                "_source": ["text", "title"],
                "from": 0, "size": 10000,
                "query": {
                    "bool" : {
                        "must": [{"match":{"text": i}} for i in name.split(' ')]
                    }
                }
            }
            r = json.loads(requests.post(self.es_host+"ebert-reviews/_search", json.dumps(q)).content.decode('utf-8', 'ignore'))
            return [Article(i['_source']['title'], i['_source']['text']) for i in r['hits']['hits']]
        else:
            raise ValueError('author must be one of "jones" or "ebert", you gave {}'.format(author))

    def retrieve_article(self, article_string):
        """Retreive a single article with a title relevant to article_string

        Args:
            article_string (string): A string with relevant keywords to an article title

        Returns:
            An Article corresponding to the article_string
        """
        if self.author == 'jones':
            q = {
                "_source": ["Full text:", "ProQ:"],
                "from": 0, "size": 1,
                "query": {
                    "bool" : {
                        "must": [{"match":{"Full text:": i}} for i in article_string.split(' ')]
                    }
                }
            }
            r = json.loads(requests.post(self.es_host+"flattened-articles/_search", json.dumps(q)).content.decode('utf-8', 'ignore'))
            return [Article(clean_article_title(i['_source']['ProQ:']), i['_source']['Full text:']) for i in r['hits']['hits']][0]
        elif self.author == 'ebert':
            q = {
                "_source": ["text", "title"],
                "from": 0, "size": 1,
                "query": {
                    "bool" : {
                        "must": [{"match":{"title": i}} for i in article_string.split(' ')]
                    }
                }
            }
            r = json.loads(requests.post(self.es_host+"ebert-reviews/_search", json.dumps(q)).content.decode('utf-8', 'ignore'))
            return [Article(i['_source']['title'], i['_source']['text']) for i in r['hits']['hits']][0]
        else:
            raise ValueError('author must be one of "jones" or "ebert", you gave {}'.format(author))


    def slurp_articles(self):
        q = {
            "size":10000,
            "query" : {
                "match_all" : {}
        }
        }
        r = json.loads(requests.post(self.es_host+"ebert-reviews/_search", json.dumps(q)).content.decode('utf-8', 'ignore'))
        return [i["_source"]["text"] for i in r['hits']['hits']]


class Article(object):
    """A class to containt an article

    Attributes:
        text (string): The full text of the article
        sentences (list of Content): The sentences of the article

    """
    def __init__(self, title, text):
        self.text = str(text.encode('utf-8', 'ignore').decode('ascii', 'ignore'))
        self.title = str(title.encode('utf-8', 'ignore').decode('ascii', 'ignore'))
        for i in ["[Tt]h(e|(is)) [Ff]ilm", "[Tt]h(e|(is)) [Mm]ovie"]:
            self.text = re.sub(i, self.title, self.text)
        self.paragraphs = self.text.splitlines()

        self.sentences = []
        for p in self.paragraphs:
            self.sentences += [Content(i, title) for i in NLP(p).sents]

    def get_paragraph(self, c):
        """Return the full paragraph that a Content, c, came from

        Args:
            c (Content): a content object

        Returns:
            A full text paragraph containing the content arg
        """

        for p in self.paragraphs:
            if re.search(c.text.text.split('(')[0], p) is not None:
                return p
        raise ValueError('Content must be in the Article, no match found for {}'.format(c.text.text))



class Content(object):
    """A lightweight class to hold content and metadata

    Attributes:
        text (string): The text
        title (string): The title of the article the text came from
        theater (string): The name of the theater the article is talking about
        director (string): The name of the director

    """

    def __init__(self, text, title=None, theater=None, director=None, names=[]):
        self.text = text
        textblob_sentiment = TextBlob(text.text).sentiment
        self.sentiment = textblob_sentiment[0]
        self.subjectivity = textblob_sentiment[1]
        self.title = title
        self.theater = theater
        self.director = director
        self.names = names
        self.filter_depth = 0



def parse_articles(article_list):
    """Parse the sentence content out of the full text of an article

    Args:
        article_list (list of tuples): a list of tuples returned from Corpus.retrieve_articles

    Returns:
        a list of content ojects, with text and title fields populated
    """
    results = []
    for i in article_list:
        for j in NLP(i[1]).sents:
            results.append(Content(j, title=i[0]))
    return results

def compose_filters(content_list, filter_list, **kwargs):

    l = content_list
    for filt in filter_list:
        tmp_l = [filt(i, **kwargs) for i in l]
        tmp_l = [i for i in tmp_l if i is not None]
        for i in tmp_l:
            i.filter_depth += 1
        if len(tmp_l) > 0:
            l = tmp_l
    return l

def filter_opinion(content, **kwargs):
    return content if content.subjectivity > 0 else None

def filter_name(content, **kwargs):
    return content if re.search(kwargs['name'], content.text.text) is not None else None

def filter_name_parentheses(content, **kwargs):
    return content if re.search("\([a-zA-Z]* *"+kwargs['name']+"\)", content.text.text) is None else None

def filter_fragments(content, **kwargs):
    return content if re.search("[a-z]", content.text.text[0]) is None \
        and re.search("[\.\?!]", content.text.text[-1]) is not None else None

def filter_stop_chars(content, **kwargs):
    return content if not any(i in content.text.text for i in ["$", "@", "- -"]) else None

def filter_person_subject_object(content, **kwargs):
    return content if len([1 for j in content.text if j.dep_ in ['nsubj', 'pobj'] and j.ent_type_ == 'PERSON']) > 0 else None

def filter_person_subject(content, **kwargs):
    return content if len([1 for j in content.text if j.dep_ in ['nsubj'] and j.ent_type_ == 'PERSON']) > 0 else None

def filter_name_subject_object(content, **kwargs):
    # return content if len([1 for j in content.text if j.dep_ in ['nsubj', 'pobj'] and j.ent_type_ == 'PERSON']) > 0 else None
    subject_object_tokens = [j.text for j in content.text if j.dep_ in ['nsubj', 'pobj']]
    name_tokens = kwargs['name'].split(' ')
    for i in name_tokens:
        if i in subject_object_tokens:
            return content
    return None


def filter_name_subject(content, **kwargs):
    subject_tokens = [j.text for j in content.text if j.dep_ in ['nsubj']]
    name_tokens = kwargs['name'].split(' ')
    for i in name_tokens:
        if i in subject_tokens:
            return content
    return None







def always_like_name(corpus, name):
    c = Corpus(corpus)
    a = c.retrieve_articles(name)
    results = []
    for article in a:
        tmp_c_l = sorted([i for i in compose_filters(article.sentences,
                                                    [filter_opinion,
                                                     filter_stop_chars,
                                                     filter_name,
                                                     filter_name_subject_object,
                                                     filter_name_subject],
                                                     name=name) if i.filter_depth >=3],
                        key=lambda x: x.sentiment,
                         reverse=False)
        if len(tmp_c_l) > 0:
            results.append((article, tmp_c_l[0]))

    sorted_results = sorted(results, key=lambda x: x[1].sentiment, reverse=False)

    for i in sorted_results:
        try:
            return i[0].get_paragraph(i[1])
        except ValueError:
            pass
    return "I'm sorry, I don't have anything to say about {}".format(name)


def always_dislike_name(corpus, name):
    c = Corpus(corpus)
    a = c.retrieve_articles(name)
    results = []
    for article in a:
        tmp_c_l = sorted([i for i in compose_filters(article.sentences,
                                                    [filter_opinion,
                                                     filter_stop_chars,
                                                     filter_name,
                                                     filter_name_subject_object,
                                                     filter_name_subject],
                                                     name=name) if i.filter_depth >=3],
                        key=lambda x: x.sentiment,
                         reverse=True)
        if len(tmp_c_l) > 0:
            results.append((article, tmp_c_l[0]))

    sorted_results = sorted(results, key=lambda x: x[1].sentiment, reverse=True)

    for i in sorted_results:
        try:
            return i[0].get_paragraph(i[1])
        except ValueError:
            pass
    return "I'm sorry, I don't have anything to say about {}".format(name)

def favorite_person_in_article(corpus, article):
    c = Corpus(corpus)
    a = c.retrieve_article(article)

    sorted_results = sorted([i for i in compose_filters(a.sentences,
                                                [filter_opinion,
                                                 filter_stop_chars,
                                                 filter_person_subject_object,
                                                 filter_person_subject]) if i.filter_depth >=3],
                    key=lambda x: x.sentiment,
                     reverse=True)

    for i in sorted_results:
        try:
            return a.get_paragraph(i)
        except ValueError:
            pass
    return "I'm sorry, I don't have anything to say about {}".format(article)


def least_favorite_in_article(corpus, article):
    c = Corpus(corpus)
    a = c.retrieve_article(article)

    results = []
    sorted_results = sorted([i for i in compose_filters(a.sentences,
                                                [filter_opinion,
                                                 filter_stop_chars,
                                                 ]) if i.filter_depth >=2],
                    key=lambda x: x.sentiment,
                            reverse=False)

    for i in sorted_results:
        try:
            return a.get_paragraph(i)
        except ValueError:
            pass
    return "I'm sorry, I don't have anything to say about {}".format(article)


def compare_person_works(corpus, name, title):
    c = Corpus(corpus)

    final_results = []
    a = c.retrieve_article(title)
    sorted_results = sorted([i for i in compose_filters(a.sentences,
                                                [filter_opinion,
                                                 filter_stop_chars,
                                                 filter_name,
                                                 filter_name_subject_object,
                                                 filter_name_subject],
                                                        name=name) if i.filter_depth >=3],
                    key=lambda x: x.subjectivity,
                     reverse=True)

    for i in sorted_results:
        try:
           final_results.append((a, i))
           break
        except ValueError:
            pass



    a2 = c.retrieve_articles(name)
    a2 = [i for i in a2 if i is not a]
    results = []
    for article in a2:
        tmp_c_l = sorted([i for i in compose_filters(article.sentences,
                                                    [filter_opinion,
                                                     filter_stop_chars,
                                                     filter_name,
                                                     filter_name_subject_object,
                                                     filter_name_subject],
                                                     name=name) if i.filter_depth >=5],
                        key=lambda x: x.subjectivity,
                         reverse=True)
        if len(tmp_c_l) > 0:
            results += [(article, i) for i in tmp_c_l]

    sorted_results = sorted(results, key=lambda x: x[1].sentiment, reverse=True)

    for i in sorted_results:
        try:
            final_results.append((i[0], i[1]))
            break
        except ValueError:
            pass
    for i in sorted_results[::-1]:
        try:
            final_results.append((i[0], i[1]))
            break
        except ValueError:
            pass

    for i in final_results:
        print(i[0].get_paragraph(i[1]))

# How do you feel about ___person___ in ___show___
def opinion_person_in_article(corpus, name, article):
    c = Corpus(corpus)
    a = c.retrieve_article(article)

    sorted_results = sorted([i for i in compose_filters(a.sentences,
                                                        [filter_opinion,
                                                         filter_stop_chars,
                                                         filter_name,
                                                         filter_name_subject_object,
                                                         filter_name_subject],
                                                        name=name) if i.filter_depth >=3],
                            key=lambda x: x.subjectivity,
                            reverse=True)

    for i in sorted_results:
        try:
            return a.get_paragraph(i)
        except ValueError:
            pass
    return "I'm sorry, I don't have anything to say about {}".format(article)




def like_person(corpus, name):
    c = Corpus(corpus)

    final_results = []

    # Get all articles that have the person's name
    article_list = c.retrieve_articles(name)
    last_name = name.split()[-1]

    filtered_sentences = []

    for article in article_list:
        filtered_sentences += [(article, i) for i in compose_filters(article.sentences,
                                       [filter_opinion,
                                        filter_person_subject,
                                        filter_name_parentheses,
                                        filter_name],
                                       name=last_name) if i.filter_depth > 3]

    print([i[1].sentiment for i in filtered_sentences])
    print(sum([i[1].sentiment for i in filtered_sentences])/ len(filtered_sentences))
    running_response = []
    for i in sorted(filtered_sentences, key=lambda x: x[1].sentiment, reverse=True):
        if i[1].filter_depth > 3:
            print(i[1].sentiment)
            print(i[1].text.text)
            try:
                print(i[0].get_paragraph(i[1]))
                running_response.append(i[0].get_paragraph(i[1]))
            except ValueError:
                print('PARAGRAPH NOT FOUND')
            print('\n')
    return "\n".join(running_response)


def like_person_2(corpus, name):
    c = Corpus(corpus)

    final_results = []

    # Get all articles that have the person's name
    article_list = c.retrieve_articles(name)

    last_name = name.split()[-1]

    filtered_sentences = []
    running_response = []
    for article in article_list:
        filtered_sentences += [(article, i) for i in compose_filters(article.sentences,
                                       [filter_opinion,
                                        filter_person_subject,
                                        filter_name_parentheses,
                                        filter_name],
                                       name=last_name) if i.filter_depth > 3]

    print([i[1].sentiment for i in filtered_sentences])
    print(sum([i[1].sentiment for i in filtered_sentences])/ len(filtered_sentences))

    ctr = 0
    sorted_results = sorted(filtered_sentences, key=lambda x: x[1].sentiment, reverse=True)
    for i in sorted_results:
        if ctr < 2:
            if i[1].filter_depth > 3:
                #print(i[1].sentiment)
                #print(i[1].text.text)
                try:
                    print(i[0].get_paragraph(i[1]))
                    running_response.append(i[0].get_paragraph(i[1]))
                    ctr += 1
                except ValueError:
                    pass#print('PARAGRAPH NOT FOUND')
                #print('\n')
    try:
        print("However, " + sorted_results[-1][0].get_paragraph(sorted_results[-1][1]))
        running_response.append("However, " + sorted_results[-1][0].get_paragraph(sorted_results[-1][1]))
    except ValueError:
        pass

    return "\n".join(running_response)

def like_person_3(corpus, name):
    c = Corpus(corpus)

    final_results = []

    # Get all articles that have the person's name
    article_list = c.retrieve_articles(name)

    last_name = name.split()[-1]

    filtered_sentences = []

    for article in article_list:
        filtered_sentences += [(article, i) for i in compose_filters(article.sentences,
                                       [filter_opinion,
                                        filter_person_subject,
                                        filter_name_parentheses,
                                        filter_fragments,
                                        filter_name],
                                       name=last_name) if i.filter_depth > 4]

    #print([i[1].sentiment for i in filtered_sentences])
    #print(sum([i[1].sentiment for i in filtered_sentences])/ len(filtered_sentences))

    ctr = 0
    sorted_results = sorted(filtered_sentences, key=lambda x: x[1].sentiment, reverse=True)

    first_last_candidates = []

    for i in sorted_results:
        if ctr < 300:
            if i[1].filter_depth > 3:
                #print(i[1].sentiment)
                #print(i[1].text.text)
                try:
                    matched_paragraph = i[0].get_paragraph(i[1])
                    paragraph_sentences = list(NLP(matched_paragraph).sents)
                    if i[1].text.text in paragraph_sentences[0].text:
                        #print('FIRST SENTENCE')
                        if i[1] not in [i[1] for i in first_last_candidates]:
                            first_last_candidates.append(i)
                    if i[1].text.text in paragraph_sentences[-1].text:
                        #pass#print('LAST SENTENCE')
                        if i[1] not in [i[1] for i in first_last_candidates]:
                            first_last_candidates.append(i)
                    #print(matched_paragraph)
                    ctr += 1
                except ValueError:
                    pass#print('PARAGRAPH NOT FOUND')
                #print('\n')
    #try:
    #    matched_paragraph = sorted_results[-1][0].get_paragraph(sorted_results[-1][1])
    #    print("However, " + matched_paragraph)
    #except ValueError:
    #    pass


    #for i in first_last_candidates:
    #    print(i[1].sentiment, i[0].title, i[1].text.text)

    grouped = {}
    for i in first_last_candidates:
        if grouped.get(i[0].title) is not None:
            grouped[i[0].title].append(i[1])
        else:
            grouped[i[0].title] = [i[1]]

    # sort groups by length
    sorted_by_length = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
    ctr = 0
    previous_average_sentiment = None
    running_response = []
    for s in sorted_by_length:
        if ctr < 4:
            curr_par = ""
            cas = sum(i.sentiment for i in s[1]) / len(s)
            if previous_average_sentiment is not None and (cas < 0 and previous_average_sentiment > 0):
                curr_par += "However, "
            previous_average_sentiment = cas
            curr_par += " ".join(i.text.text for i in s[1])
            ctr += 1
            running_response.append(curr_par)

    return "\n".join(running_response)




def like_person_4(corpus, name):
    c = Corpus(corpus)

    final_results = []

    # Get all articles that have the person's name
    article_list = c.retrieve_articles(name)

    last_name = name.split()[-1]

    filtered_sentences = []

    for article in article_list:
        filtered_sentences += [(article, i) for i in compose_filters(article.sentences,
                                       [filter_opinion,
                                        filter_person_subject,
                                        filter_name_parentheses,
                                        filter_fragments,
                                        filter_name],
                                       name=last_name) if i.filter_depth > 4]

    ctr = 0
    sorted_results = sorted(filtered_sentences, key=lambda x: x[1].sentiment, reverse=True)

    first_last_candidates = []

    for i in sorted_results:
        if ctr < 300:
            if i[1].filter_depth > 0:

                try:
                    matched_paragraph = i[0].get_paragraph(i[1])
                    paragraph_sentences = list(NLP(matched_paragraph).sents)
                    if i[1].text.text in paragraph_sentences[0].text:
                        #print('FIRST SENTENCE')
                        if i[1] not in [j[1] for j in first_last_candidates]:
                            tmp = i
                            tmp[1].filter_depth = tmp[1].filter_depth * 1.5
                            first_last_candidates.append(tmp)
                    elif i[1].text.text in paragraph_sentences[-1].text:
                        #pass#print('LAST SENTENCE')
                        if i[1] not in [j[1] for j in first_last_candidates]:
                            tmp = i
                            tmp[1].filter_depth = tmp[1].filter_depth * 1.5
                            first_last_candidates.append(tmp)
                    #print(matched_paragraph)
                    else:
                        first_last_candidates.append(i)
                    ctr += 1
                except ValueError:
                    pass#print('PARAGRAPH NOT FOUND')
                #print('\n')

    grouped = {}
    for i in first_last_candidates:

        if any(j in i[1].text.text.lower() for j in ['here', 'it']):
               i[1].filter_depth = i[1].filter_depth * 0.9

        if grouped.get(i[0].title) is not None:
            grouped[i[0].title].append(i[1])
        else:
            grouped[i[0].title] = [i[1]]

    for k in grouped:

        v = sorted(grouped[k], key=lambda x: x.filter_depth, reverse=True)
        avg_sim = []
        for i in range(len(v)):
            for j in range(1, len(v)):
                avg_sim += [v[i].text.similarity(v[j].text)]
        if len(avg_sim) > 0:
            avg_sim = sum(avg_sim)/len(avg_sim)
        else:
            avg_sim = 0.0
        for i in v:
            i.filter_depth = i.filter_depth * avg_sim

        grouped[k] = v


    # sort groups by score
    sorted_by_length = sorted(grouped.items(), key=lambda x: max(a.filter_depth for a in x[1]), reverse=True)
    ctr = 0
    previous_average_sentiment = None
    running_response = []
    for s in sorted_by_length:
        if ctr < 4:
            curr_par = ""
            cas = sum(i.sentiment for i in s[1]) / len(s)
            if previous_average_sentiment is not None and (cas < 0 and previous_average_sentiment > 0):
                curr_par += "However, "
            previous_average_sentiment = cas
            curr_par += " ".join(i.text.text for i in s[1])
            ctr += 1
            running_response.append(curr_par)

    return "\n".join(running_response)


def anaphor_overlap(summary_text):
    """This function computes the anaphor overlap index

    Per the Coh-Metrix Documentation:

        This measure considers the anphor overlap between pairs of
        sentences. A pair of sentences has an anphor overlap if the
        later sentence contains a pronoun that refers to a pronoun
        or noun in the earlier sentence. The score for each pair of
        sentences is binary, i.e., 0 or 1. The measure of the text
        is the average of the pair scores.

    Args:
        summary_text (String): The text to be scored

    Return:
        Float - the anaphor overlap score
    """

    # Analyze Text w/ Spacy
    annotated_text = NLP(summary_text)
    sentences = list(annotated_text.sents)

    # Iterate through sentences
    num_anaphors = 0
    for i in range(1, len(sentences)):
        if sentences[i - 1].text not in [' ', '', '\n'] and sentences[i].text not in [' ', '', '\n']:
            # check if prounoun in i and either noun or matching pronoun in i-1

            # First, check if there is a person in i-1
            person_flag = False
            for word in sentences[i-1]:
                if word.ent_type_ == 'PERSON':
                    person_flag = True

            # Next, check if there is a pronoun in i-1
            pronouns = ['he', 'she', 'they', 'him', 'her', 'them', 'his', 'hers', 'theirs']
            pronoun_flag = any(j in sentences[i-1].text.lower() for j in pronouns)

            # Check if there is a pronoun in i
            pronoun_flag_2 = any(j in sentences[i].text.lower() for j in pronouns)

            if (pronoun_flag_2 and pronoun_flag) or (pronoun_flag_2 and person_flag):
                num_anaphors += 1

    return num_anaphors / (len(sentences) - 1)


def person_overlap(summary_text):
    """This function computes the person overlap index

    Roughly based on the Noun Overlap index from Coh-Metrix

    Args:
        summary_text (String): The text to be scored

    Return:
        Float - the person overlap score
    """

    # Analyze Text w/ Spacy
    annotated_text = NLP(summary_text)
    sentences = list(annotated_text.sents)

    # Iterate through sentences
    num_anaphors = 0
    for i in range(1, len(sentences)):
        if sentences[i - 1].text not in [' ', '', '\n'] and sentences[i].text not in [' ', '', '\n']:
            # check if name in i and in i-1
            people_match = False

            # First, check if there is a person in i-1
            people = []
            for word in sentences[i-1]:
                if word.ent_type_ == 'PERSON':
                    people.append(word.text.lower())

            for word in sentences[i]:
                if word.ent_type_ == 'Person':
                    if word.text.lower() in people:
                        people_match = True

            if people_match:
                num_anaphors += 1


    return num_anaphors / (len(sentences) - 1)


EBERT_READABILITY = 9.782916286401726
def score_summary_2(summary_text):
    """Score a summarized piece of text
    """
    # Want high similarity between paragraphs
    inter_paragraph_similarities = []
    avg_similarity = None

    sentences = [i.text for i in NLP(summary_text).sents]

    # readability measures close to ebert baseline
    readability = abs(text_stats.TextStats(NLP(summary_text)).automated_readability_index - EBERT_READABILITY)/EBERT_READABILITY


    # Coh Metrix Indices
    anaphor_score = anaphor_overlap(summary_text)
    person_score = person_overlap(summary_text)


    # more subjective is better
    total_subjectivity = 0
    for i in sentences:
        total_subjectivity += TextBlob(i).sentiment[1]
    subjectivity = total_subjectivity/len(sentences)

    # thesis sentence doesn't have "this", "here", "it"
    if sentences[0] not in [' ', '', '\n']:
        thesis_penalty = sum(i in sentences[0] for i in [" this ", " This ", " here ", " Here"])
    elif sentences[1] not in [' ', '', '\n']:
        thesis_penalty = sum(i in sentences[1] for i in [" this ", " This ", " here ", " Here"])
    else:
        thesis_penalty = 0

    # Prefer expressions from the author
    author_count = 0
    for s in sentences:
        if any(i in s for i in ["I ", "I'd", "My"]):
            author_count += 1

    # iterate through the paragraphs
    # sentiment within a paragraph is similar
    paragraphs = summary_text.split('\n')
    for i in range(1, len(paragraphs)):
        if paragraphs[i - 1] not in [' ', '', '\n'] and paragraphs[i] not in [' ', '', '\n']:
            inter_paragraph_similarities.append(similarity.word_movers(NLP(paragraphs[i - 1]), NLP(paragraphs[i])))

    max_diff = 0
    for p in paragraphs:
        p_sent_min = None
        p_sent_max = None
        for s in p.split('.'):
            sent = TextBlob(s).sentiment[0]
            if p_sent_min is None:
                p_sent_min = sent
            if p_sent_max is None:
                p_sent_max = sent

            if sent < p_sent_min:
                p_sent_min = sent
            if sent > p_sent_max:
                p_sent_max = sent
        if max_diff < abs(p_sent_max - p_sent_min):
            max_diff = abs(p_sent_max - p_sent_min)
    max_diff = 1 - max_diff
    avg_similarity = sum(inter_paragraph_similarities)/len(inter_paragraph_similarities)



    # Make score
    score = (0.25 * avg_similarity) + \
            (0.20 * person_score) + \
            (0.15 * anaphor_score) + \
            (0.1 * max_diff) + \
            (0.05 * readability) + \
            (0.25 * subjectivity)
    # boost by person count
    score = score * (1 + (0.1 * author_count))
    score = score - (0.2 * thesis_penalty)


    return score

def like_person_wrapper(corpus, name):
    """This function evaluates a few handlers and selects the response with teh best fit score

    Args:
        corpus - the name of the corpus , either 'ebert', or 'jones'
        name - the name of the person

    Return:
        STring - a few paragraphs about the person

    """
    results = []
    for f in [like_person_3]:#[like_person, like_person_2, like_person_3, like_person_4]:
        results.append(f(corpus, name))

    max_score = 0
    result = None
    for r in results:
        s = score_summary_2(r)
        print(s)
        if s >= max_score:
            max_score = s
            result = r

    return result
