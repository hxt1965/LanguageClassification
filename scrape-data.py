# scrapes data from English and Dutch Wikipedia for training the data model
#
# Usage: python3 scrape-data.py [nl|en]

import requests
from bs4 import BeautifulSoup
import re
import random
import sys

# URLs for English and Dutch
EN_WIKI = 'https://en.wikipedia.org/wiki/Special:Random'
NL_WIKI = 'https://nl.wikipedia.org/wiki/Speciaal:Willekeurig'

url = ''
country = ''
if sys.argv[1].lower() == 'nl':
    url = NL_WIKI
    country = 'nl'
elif sys.argv[1].lower() == 'en':
    url = EN_WIKI
    country = 'en'
else:
    print('Usage: py scrap-data.py [en/nl]', file=sys.stderr)
    quit()

with open('data_en.txt' if country == 'en' else 'data_nl.txt',mode='a') as f:
    count = 0
    while count < 500:
        # printing to stderr doesn't redirect into file
        print(count, file=sys.stderr)

        raw_html = requests.get(url).text
        bs = BeautifulSoup(raw_html, 'html.parser')

        #print(bs.find_all('p', text=True, recursive=True))

        def clean_text(text):
            """Strip out all tags and other needless info from the text"""
            text = re.sub(r'<\\?.*?>', '', text)
            text = re.sub(r'\[\d+\]', '', text)
            text = re.sub(r'\n', ' ', text)
            #text = re.sub(r'\(.*?\)', '', text) # remove parenthesis
            #text = re.sub(r'[\'",:]', '', text) # remove punctuation
            #text = re.sub(r'\d+(\.\d+)?', '', text) # remove numbers
            
            return text

        # All article content that we care about on Wikipedia is stored in P tags, and
        # they are the only content stored in P tags, which is convienient for us.


        all_text = list(map(lambda x: clean_text(str(x)), bs.select('p')))
        all_text = list(filter(lambda x: x != '', all_text))
        #all_text = ''.join(all_text)
        # all_sentences = list(filter(lambda x: x.strip() != '', ''.join(all_text).split('.')))
        # all_sentences = list(map(lambda x: x.strip(), all_sentences))
        paragraphs = list(map(lambda x: x.split(), all_text))

        sentences = [] # list of sentences to be added to traning

        for p in paragraphs:
            # p is a list of strings (words)
            if len(p) < 15:
                continue
            
            low = random.randint(0, len(p) - 15)
            high = low + 15

            count += 1
            f.write(f"{country}|{' '.join(p[low:high])}\n")

        #print(words)