##PERFORMING BASIC NLP (natural language processing)

import nltk
text="my name is ganesh. i recently from py hey! am an the completed my graduation in bba business analytics. at university post graduate college"
# above is a dummy text to perform NLP operations

# 1. make it all same case (lower case preferably)
text1=text.lower()
print("printing lower text \n",text1)

#senetence segmentation breaking into sentence

from nltk.tokenize import sent_tokenize,word_tokenize
text2=sent_tokenize(text)
print("printing sent_tokenize \n",text2)
text3=word_tokenize(text)
print("printing word_tokenize \n",text3)

# removing stop words
my_stop_words={"is","in","at","my","i"}
text4=text3
for i in list(text3):
    if i in my_stop_words:
        count_i=text3.count(i)
        for j in range(count_i):
            text3.remove(i)
print("printing removing stop words \n",text4)


text5=text3
from nltk.corpus import stopwords


# performing using inbuilt features
nltk_stop_words=set(stopwords.words("english"))
print("printing stop words \n",nltk_stop_words)
nltk_stop_words.remove("my")

for i in list(text5):
    if i in nltk_stop_words:
        count_i=text5.count(i)
        for j in range(count_i):
            text5.remove(i)
print("printing nltk stop words \n",text5)


# find the root words - Stemming techniques
# most popular stemming technique is Porter stemming
# another one we have word net lemmatizer we discuss it later

from nltk.stem import PorterStemmer
stmmer=PorterStemmer()
text6=[stmmer.stem(word)for word in text5]
print("printing nltk steming\n",text6)

# Tagging parts of speech - POS
# tagger class of nltk will help us to tag the words

from nltk.tag import DefaultTagger
tagger=DefaultTagger('NN')
tag_txt=tagger.tag(text5)
print("printing tagger \n",tag_txt)


# let's find out POS of each word from the text
nltk.download("averaged_perceptron_tagger_eng")

print("printing each word tag of pos\n",nltk.pos_tag(text5))
