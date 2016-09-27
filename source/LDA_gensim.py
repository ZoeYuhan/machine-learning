#!/usr/bin/python
#-*- coding:utf-8 -*-

from gensim import corpora,model,similarities
from pprint import pprint

if __name__=='__main__':
    f=open('test_txt')
    stop_list=set('for a of the and to in'.split())
    texts=[[word for word in line.strip().lower.split() if word not in stop_list] for line in f]
    print 'Text= '
    pprint(texts)
    
    dictionary=corpora.Dictionary(texts)
    V=len(dictionary)
    corpus[dictionary.doc2bow(text) for text in texts]
    corpus_tfidf=models.TfidfModel(corpus)[corpus]
    
    print 'TF-IDF:'
    for c in corpus_tfidf:
        print c
    
    print '\nLSI Model:'
    lsi=models.LsiModel(corpus_tfidf,num_topic=2,id2word=dictionary)
    topic_result=[a for a in lsi[corpus_tfidf]]
    pprint(topic_result)
    print 'LSI Topics:'
    pprint(lsi.print_topics(num_topics=2,num_words=5))
    similarity=similarities.MatrixSimilarity(lsi[corpus_tfidf])
    print 'Similarity:'
    pprint(list(similarity))
    
    print '\nLDA Model:'
    num_topic=2
    lad=models.LdaModel(corpus_tfidf,num_topics=num_topics,id2word=dictionary,alpha='auto',eta='auto',minimum_probability=0.01)
    doc_topic=[a for a in lda[corpus_tfidf]]
    print 'Document-Topic:\n'
    pprint(doc_topic)
    
    for topic_id in range(num_topics):
        print 'Topic', topic_id
        pprint(lda.show_topic(topic_id))
    similarity=similarities.MaxtrixSimilarity(lda[corpus_tfidf])
    print'Similarity:'
    pprint (list(similarity))
    
