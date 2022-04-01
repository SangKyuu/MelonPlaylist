# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:18:54 2020

@author: KE
"""


import os, sys
import json
import pandas as pd
from tqdm import tqdm

from arena_util import write_json
from arena_util import remove_seen
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from gensim.models import KeyedVectors


from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def wtf(a):
    print(type(a))
    try: print(a.shape)
    except: print(len(a))
    
    
class PlaylistEmbedding:
    def __init__(self, FILE_PATH):
        self.FILE_PATH = FILE_PATH
        self.min_count = 3
        self.size = 300
        self.window = 210
        self.sg = 5
        self.p2v_model = WordEmbeddingsKeyedVectors(300)

        with open(os.path.join(FILE_PATH, 'train.json'), encoding="utf-8") as f:
            self.train = json.load(f)
        with open(os.path.join(FILE_PATH, 'val.json'), encoding="utf-8") as f:
            self.val = json.load(f)
        with open(os.path.join(FILE_PATH, 'results.json'), encoding="utf-8") as f:
            self.most_results = json.load(f)
        with open(os.path.join(FILE_PATH, 'song_meta.json'), encoding='utf-8') as f:
            self.song_meta = json.load(f)
        with open(os.path.join(FILE_PATH, 'genre_gn_all.json'), encoding='utf-8') as f:
            self.gnr_all = json.load(f)

    def _genre(self, songs):
        gnrs = list(y for x in songs for y in self.song_meta[x]['song_gn_gnr_basket'])
        def _get_genre(genre):
            try:
                return self.gnr_all[genre]
            except KeyError:
                pass
        gnrs = list(set(map(_get_genre, gnrs)))
        if None in gnrs:
            gnrs.remove(None)
        return gnrs

    def _get_songmeta(self, song):
        songmeta = self.song_meta[song]
        try:
            songmeta = ' '.join(songmeta['song_gn_dtl_gnr_basket'] + songmeta['song_gn_gnr_basket'] +
                                [songmeta['album_name']] + [songmeta['song_name']] +
                                songmeta['artist_name_basket'] + list(map(str, songmeta['artist_id_basket'])) +
                                [str(songmeta['id'])])
            return songmeta
        except:
            return ''

    def get_dic(self, train, val):
        song_dic = {}
        tag_dic = {}
        data = train + val
        total = []
        for q in tqdm(data):
            song_dic[str(q['id'])] = q['songs']
            tag_dic[str(q['id'])] = q['tags']
            p1 = ' '.join(q['tags'] + q['plylst_title'].split() + list(map(str, q['songs'])))
            p2 = ' '.join(map(self._get_songmeta, q['songs']))
            total.append({'sentence':p1 + ' ' + p2, 'id':q['id']})

        self.song_dic = song_dic
        self.tag_dic = tag_dic

        total = [x for x in total if len(x) > 1]
        self.total = total

    def get_song2vec(self):
        max_epochs = 5
        vec_size = 300
        alpha = 0.025
        
        try:
            s2v_model = Doc2Vec.load('./s2v_model.model')
            self.s2v_model = s2v_model
            print("s2v_model load")
        except:
            print("create s2v_model")
            # twitter = Twitter()
            # twitter.nouns('가나다 암이 하아 고호')
            tagged_data = [TaggedDocument(words=dict['sentence'].lower().split(), tags=[str(dict['id'])]) for dict in self.total]

            s2v_model = Doc2Vec(size = vec_size, alpha=alpha, min_alpha=0.00025, min_count=3, dm =1, workers=4)
            s2v_model.build_vocab(tagged_data)
            
            for epoch in tqdm(range(max_epochs)):
                print('iteration {}'.format(epoch))
                s2v_model.train(tagged_data, total_examples = s2v_model.corpus_count, epochs = s2v_model.iter)
                # decrease the learning rate
                s2v_model.alpha -= 0.0002
                # fix the learning rate, no decay
                s2v_model.min_alpha = s2v_model.alpha
                
            s2v_model.save("./s2v_model.model")
            print("save s2v_model model")
            self.s2v_model = s2v_model

    def get_result(self, song_dic, tag_dic, most_results, val):
        answers = []
        for n, q in tqdm(enumerate(val), total = len(val)):
            # tag
            try:
                most_id = [x[0] for x in self.s2v_model.docvecs.most_similar(str(q['id']), topn=200)]
                get_tag = []
                get_song = []
                for ID in most_id:
                    get_tag += tag_dic[ID]
                    get_song += song_dic[ID]
                get_song = list(pd.value_counts(get_song)[:200].index)
                get_tag = list(pd.value_counts(get_tag)[:20].index)
                # appned
                answers.append({
                    "id": q["id"],
                    "songs": remove_seen(q["songs"], get_song)[:100],
                    "tags": remove_seen(q["tags"], get_tag)[:10],
                })            # tag except
            except:
                answers.append({
                    "id": most_results[n]["id"],
                    "songs": most_results[n]['songs'],
                    "tags": most_results[n]["tags"],
                })

        for n, q in enumerate(answers):
            if len(q['songs']) != 100:
                answers[n]['songs'] += remove_seen(q['songs'], self.most_results[n]['songs'])[:100 - len(q['songs'])]
            if len(q['tags']) != 10:
                answers[n]['tags'] += remove_seen(q['tags'], self.most_results[n]['tags'])[:10 - len(q['tags'])]
        self.answers = answers
       

    def run(self):
        self.get_dic(self.train, self.val)
        
        self.get_song2vec()

        self.get_result(self.song_dic, self.tag_dic, self.most_results, self.val)
        write_json(self.answers, 'results.json')


#%%

FILE_PATH = './arena_data'
U_space = PlaylistEmbedding(FILE_PATH)
U_space.run()












