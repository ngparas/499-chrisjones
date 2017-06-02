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



class Article(object):
    """A class to containt an article

    Attributes:
        text (string): The full text of the article
        sentences (list of Content): The sentences of the article

    """
    def __init__(self, title, text):
        self.text = str(text.encode('utf-8', 'ignore').decode('ascii', 'ignore'))
        self.title = str(title.encode('utf-8', 'ignore').decode('ascii', 'ignore'))
        for i in ["th(e|(is)) [Ff]ilm", "th(e|(is)) [Mm]ovie"]:
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

def main():
    #    print('\n')
    #    print('Roger Ebert, do you always like Francis Ford Coppola?')
    #    print(always_like_name('ebert', 'Francis Ford Coppola'))
    #    print('\n')
    #    print('Roger Ebert, do you always like Viggo Mortensen?')
    #    print(always_like_name('ebert', 'Viggo Mortensen'))
    #    print('\n')
    #    print('Chris Jones, do you always like Aaron Sorkin?')
    #    print(always_like_name('jones', 'Aaron Sorkin'))
    #    print('\n')
    #    print('Chris Jones, do you always like Aaron Todd Douglas?')
    #    print(always_like_name('jones', 'Aaron Todd Douglas'))
    #    print('\n')
    #    print('Chris Jones, do you always dislike Aaron Todd Douglas?')
    #    print(always_dislike_name('jones', 'Aaron Todd Douglas'))
    #    print('\n')
    #    print('Chris Jones, do you always like Tracy Letts?')
    #    print(always_like_name('jones', 'Tracy Letts'))
    #    print('\n')
    #    print('Chris Jones, do you always dislike Steppenwolf?')
    #    print(always_dislike_name('jones', 'Steppenwolf'))

    #    print('\n')
    #    print('Roger Ebert, who was your favorite person in Apocalypse Now?')
    #    print(favorite_person_in_article('ebert', 'Apocalypse Now'))
    #    print('\n')
    #    print('Chris Jones, who was your favorite person in Killer Joe?')
    #    print(favorite_person_in_article('jones', 'Killer Joe'))
    #    print('\n')
    #    print('Roger Ebert, what was the worst thing in Apocalypse Now Redux?')
    #    print(least_favorite_in_article('ebert', 'Apocalypse Now Redux'))

    #    print('\n')
    #    # One of the original questions that started this endeavor was
    #    # "Chris, well doy just always dislike mexican directors?". The approach
    #    # that I favor is to actually use an external databse, knowledge graph, etc.
    #    # to get a list of related directors according to whatever the characteristic
    #    # in question may be (in this case, country of origin), and then execute the
    #    # same dislike_name function we've been calling before for each of them.

    #    # Shockingly, if we call it for Alfonso Cuaron (who would very likely be
    #    # near the top of any list of prominent Mexican filmmakers, especially
    #    # after Children of Men and Gravity), we get the following:

    #    # print('Roger Ebert, do you always dislike Alejandro Gonzalez Inarritu?')
    #    # print(always_dislike_name('ebert', 'Alejandro Gonzalez Inarritu'))
    #    print('Roger Ebert, do you always dislike Mexican Filmmakers?')
    #    print(always_dislike_name('ebert', 'Alfonso Cuaron'))

    #    print('\n')
    #    compare_person_works('jones', 'Tracy Letts', 'August Osage County')

    #    print('\n')
    #    compare_person_works('jones', 'Steppenwolf Theatre', 'Buried Child')

    print('\n')
    print('Ebert, what did you think of Jennifer Lawrence in Winters Bone?')
    print(opinion_person_in_article('ebert', 'Jennifer Lawrence', "Winter's Bone"))

    print('\n')
    print('Jones, what did you think of Tom Bateman in AmeriKafka?')
    print(opinion_person_in_article('jones', 'Bateman', "AmeriKafka"))


    c = Corpus('jones')
    arts = c.filter_articles_by_person("Bateman's")
    for a in arts:
        print(a.title)
if __name__ == '__main__':
    main()
