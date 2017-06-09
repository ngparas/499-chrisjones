import os
import time
from slackclient import SlackClient
# from chrisjones import ChrisJones
from collections import deque
import string
import spacy

nlp = spacy.load('en')

from new_query_handlers import *

# constants
READ_WEBSOCKET_DELAY = 1 # 1 second delay between reading from firehose

class SlackBot:
    def __init__(self, token, bot_id):
        self.bot_id = bot_id
        self.bot_tag = '<@' + bot_id + '>'
        self.client = SlackClient(token)
        # self.controller = ChrisJones()
        self.memory = deque()

    def listen(self):
        if self.client.rtm_connect():
            print("StarterBot connected and running!")
            while True:
                command, channel = self.parse_slack_output(self.client.rtm_read())
                if command and channel:
                    self.handle_command(command, channel)
                    time.sleep(READ_WEBSOCKET_DELAY)
        else:
            print("Connection failed. Invalid Slack token or bot ID?")

    def handle_command(self, command, channel):
        """
            Receives commands directed at the bot and determines if they
            are valid commands. If so, then acts on the commands. If not,
            returns back what it needs for clarification.
        """

        query_type = get_question_type(command)

        response = self.generate_response(command, query_type)
        self.client.api_call("chat.postMessage", channel=channel, text=response, as_user=True)

    def parse_slack_output(self, slack_rtm_output):
        """
            The Slack Real Time Messaging API is an events firehose.
            this parsing function returns None unless a message is
            directed at the Bot, based on its ID.
        """
        output_list = slack_rtm_output
        if output_list and len(output_list) > 0:
            for output in output_list:
                if output and 'text' in output and self.bot_tag in output['text']:
                    # return text after the @ mention, whitespace removed
                    # msg = output['text'].split(self.bot_tag)[1].encode('utf8').translate(None, string.punctuation).strip()
                    msg = output['text'].split(self.bot_tag)[1].strip()
                    return msg, output['channel']
        return None, None

    def generate_response(self, query_text, query_type):

        # Grab People to Start
        annotated_query = nlp(query_text)
        people = [i.text for i in annotated_query if i.ent_type_ == "PERSON"]
        if len(people) <= 3 and len(people) > 0:
            person = " ".join(people)
        else:
            # Get previous person
            if any(i in query_text.lower() for i in ["him", "her", "them"]):
                person = self.memory[-1].get('person')
            else:
                person = None

        if query_type is None:
            print('using previous context for query type')
            query_type = self.memory[-1].get('query_type')



        print('{} question type for person: {}'.format(query_type, person))

        if query_type == 'favorite_person_in_article':
            response = favorite_person_in_article('ebert', query_text)
        elif query_type == "least_favorite_in_article":
            response = least_favorite_in_article('ebert', query_text)

        elif person is not None:

            if query_type == "compare_person_works":
                if not ("in it" in query_text):
                    response = compare_person_works('ebert', person, " ".join(query_text.split(' ')[-2:]))
                    film_title = " ".join(query_text.split(' ')[-2:])
                else:
                    film_title = self.memory[-1].get('film_title')
                    response = compare_person_works('ebert', person, film_title)
            elif query_type == 'opinion_person_in_article':
                if not ("in it" in query_text):
                    response = opinion_person_in_article('ebert', person, " ".join(query_text.split(' ')[-2:]))
                    film_title = " ".join(query_text.split(' ')[-2:])
                else:
                    film_title = self.memory[-1].get('film_title')
                    print('Grabbed film title from stack, {}'.format(film_title))
                    response = opinion_person_in_article('ebert', person, film_title)
            elif query_type == "always_dislike_name":
                response = always_dislike_name('ebert', person)
                film_title = None
            elif query_type == 'always_like_name':
                response = always_like_name('ebert', person)
                film_title = None
            elif query_type == 'like_person':
                response = like_person_wrapper('ebert', person)
                film_title = None
        else:
            response = "No person found, more question support coming soon"
            film_title = None




        #response = "You asked about {}".format(person)


        self.memory.append({
            'query_text': query_text,
            'query_type': query_type,
            'film_title': film_title,
            'person': person,
            'response': response
        })

        print(film_title)

        return response


def get_question_type(query_text):
    # Options are:
    # always like
    # always dislike
    # favorite in article
    # least favorite in article
    # compare a psersons works
    # opinion of person in an article
    # like person (multiple handlers and a wrapper)

    if "compare" in query_text:
        return "compare_person_works"
    elif "least" in query_text:
        return "least_favorite_in_article"
    elif "think" in query_text:
        return "opinion_person_in_article"
    elif "favorite" in query_text:
        return "favorite_person_in_article"
    elif "always" in query_text:
        if "dislike" in query_text:
            return "always_dislike_name"
        else:
            return "always_like_name"
    elif "like" in query_text:
        return "like_person"
    else:
        return None


if __name__ == "__main__":
    bot = SlackBot(os.environ.get('SLACK_BOT_TOKEN'), os.environ.get('SLACK_BOT_ID'))
    bot.listen()
