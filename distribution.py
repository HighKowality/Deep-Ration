import json
import sys
import nltk
from nltk.probability import ConditionalFreqDist


def main():
    post_dict = {}
    authors = []
    cfdist = ConditionalFreqDist()

    with open(sys.argv[1]) as jf:
        jd = json.load(jf)

    for x in range(0, len(jd['data'])):
        tok = nltk.word_tokenize(jd.get('data')[x].get('data').get('selftext').get('title'))
        authors.append(jd.get('data')[x].get('data').get('author_fullname'))
        post_dict[jd.get('data')[x].get('data').get('title')] = nltk.pos_tag(tok)

    with open("output.txt", 'w') as graph:
        for x, y in post_dict.items():

            condition = post_dict[jd.get('data')[x].get('data').get('title')]
            cfdist[condition][y] += 1

            graph.nltk.plot()


if __name__ == '__main__':

    main()
