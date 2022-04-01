import os
import json

import pandas as pd

from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from arena_util import write_json
from arena_util import remove_seen
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors


# rev20 : title / song 으로 나누어 p2v를 2건 만드는 버전

class PlaylistEmbedding:
    def __init__(self, FILE_PATH):
        self.FILE_PATH = FILE_PATH
        self.min_count = 1
        self.size = 300
        self.window = 210
        self.sg = 5
        self.p2v_model = WordEmbeddingsKeyedVectors(self.size)

        with open(os.path.join(FILE_PATH, 'train.json'), encoding="utf-8") as f:
            self.train = json.load(f)
        with open(os.path.join(FILE_PATH, 'val.json'), encoding="utf-8") as f:
            self.val = json.load(f)
        with open(os.path.join(FILE_PATH, 'test.json'), encoding="utf-8") as f:
            self.test = json.load(f)                 
        with open(os.path.join(FILE_PATH, 'results.json'), encoding="utf-8") as f:
            self.most_results = json.load(f)
        with open(os.path.join(FILE_PATH, 'song_meta.json'), encoding='utf-8') as f:
            self.song_meta = json.load(f)
        with open(os.path.join(FILE_PATH, 'genre_gn_all.json'), encoding='utf-8') as f:
            self.gnr_all = json.load(f)
        
        self.total_data = self.train + self.val + self.test
        
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

    def get_song_title(self, song):
        return self.song_meta[song]['song_name']

    def get_dic(self, total_data, song_meta):
        print('get_dic')
        song_dic = {}
        tag_dic = {}
        data = total_data
        for q in tqdm(data):
            song_dic[str(q['id'])] = q['songs']
            tag_dic[str(q['id'])] = q['tags']
        self.song_dic = song_dic
        self.tag_dic = tag_dic
        # total = list(map(lambda x: list(x['tags']) + [x['plylst_title']] + self._genre(x['songs']), data))
        total_data_split = list(map(lambda x: list(x['tags']) + x['plylst_title'].split() + list(map(str, x['songs'])), data))

        total_data_split = [x for x in total_data_split if len(x) > 1]
        self.total_data_split = total_data_split

    def get_w2v(self, total_data_split, min_count, size, window, sg):
        try:
            print('\n load w2v')
            w2v_model = Word2Vec.load('./w2v_train_val_test_300_min1.model')
            self.w2v_model = w2v_model
            print('\n load success')
        except:
            print('\n create w2v')
            w2v_model = Word2Vec(total_data_split, min_count=min_count, size=size, window=window, sg=sg)
            w2v_model.save('./w2v_train_val_test_300_min1.model')
            self.w2v_model = w2v_model


    def update_p2v(self, total_data, w2v_model):
        print('\n update_p2v')
        ID = []
        vec = []
        for q in tqdm(total_data):
            tmp_vec = 0
            for word in q['tags'] + q['plylst_title'].split() + list(map(str, q['songs'])):
                try:
                    tmp_vec += w2v_model.wv.get_vector(str(word))
                except KeyError:
                    pass
            if type(tmp_vec) != int:
                ID.append(str(q['id']))
                vec.append(tmp_vec)
        self.p2v_model.add(ID, vec)


    def get_result(self, p2v_model, song_dic, tag_dic, most_results, val):
        print('\n get_result')
        answers = []
        for n, q in tqdm(enumerate(self.test), total = len(self.test)):
            # try:
            #     # tag
            #     most_id = [x[0] for x in p2v_model.most_similar(str(q['id']), topn=200) ]
            #     get_song = []
            #     get_tag = []
            #     for ID in most_id:
            #         get_song += song_dic[ID]
            #         get_tag += tag_dic[ID]
            #     get_song = list(pd.value_counts(get_song)[:200].index)
            #     get_tag = list(pd.value_counts(get_tag)[:20].index)
            #     answers.append({
            #         "id": q["id"],
            #         "songs": remove_seen(q["songs"], get_song)[:100],
            #         "tags": remove_seen(q["tags"], get_tag)[:10],
            #     })
            # except:
            #     answers.append({
            #         "id": most_results[n]["id"],
            #         "songs": most_results[n]['songs'],
            #         "tags": most_results[n]["tags"],
            #     })

            # song
            try:
                top_id = [x for x in p2v_model.most_similar(str(q['id']), topn=200)]
                
                # most_song_id = [x[0] for x in p2v_model.most_similar(str(q['id']), topn=200) if x[1]>0.9]
                most_song_id = [i[0] for i in top_id if i[1]>0.9]
                get_song = []
                for ID in most_song_id:
                    get_song += song_dic[ID]
                get_song = list(pd.value_counts(get_song)[:200].index)
                # appned
                answers.append({"id": q["id"],
                                "songs": remove_seen(q["songs"], get_song)[:100]})
            # song except
            except:
                answers.append({ "id": most_results[n]["id"],
                                 "songs": most_results[n]['songs']})                

            # tag
            try:
                most_id = [i[0] for i in top_id ]
                get_tag = []
                for ID in most_id:
                    get_tag += tag_dic[ID]
                get_tag = list(pd.value_counts(get_tag)[:20].index)
                # appned
                answers[-1]['tags'] = remove_seen(q["tags"], get_tag)[:10]
            # tag except                
            except:
                answers[-1]['tags'] = most_results[n]["tags"]

        # check and update answer
        for n, q in enumerate(answers):
            if len(q['songs'])!=100:
                answers[n]['songs'] += remove_seen(q['songs'], self.most_results[n]['songs'])[:100-len(q['songs'])]
            if len(q['tags'])!=10:
                answers[n]['tags'] += remove_seen(q['tags'], self.most_results[n]['tags'])[:10-len(q['tags'])]
        self.answers = answers
        assert len(self.answers) == len(self.test)

    def run(self):
        self.get_dic(self.total_data, self.song_meta)
        self.get_w2v(self.total_data_split, self.min_count, self.size, self.window, self.sg)
        self.update_p2v(self.total_data, self.w2v_model)
        self.get_result(self.p2v_model, self.song_dic, self.tag_dic, self.most_results, self.val)
        write_json(self.answers, 'results.json')


FILE_PATH = './arena_data/'
U_space = PlaylistEmbedding(FILE_PATH)
U_space.run()