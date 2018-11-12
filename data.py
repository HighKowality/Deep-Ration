import json
import sys
import nltk


def main():
    count = 0
    post_dict = {}
    authors = []

    with open(sys.argv[1]) as jf:
        jd = json.load(jf)

    for x in range(0, len(jd['data'])):
        tok = nltk.word_tokenize(jd.get('data')[x].get('data').get('selftext'))
        authors.append(jd.get('data')[x].get('data').get('author_fullname'))
        post_dict[jd.get('data')[x].get('data').get('title')] = nltk.pos_tag(tok)

    with open("dist.txt", 'w') as of:
        for x, y in post_dict.items():

            of.write('______________________________________________________________________________\n')
            of.write("Title:" + " " + x + "\n\n")

            of.write(', '.join(map(str, y)) + "\n")

            of.write('______________________________________________________________________________\n\n\n\n')
            # count += 1


if __name__ == '__main__':

    main()
