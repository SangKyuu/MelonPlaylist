import os
import json

import pandas as pd

from tqdm import tqdm
from arena_util import write_json
from arena_util import remove_seen
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors


class PlaylistEmbedding:
    def __init__(self, FILE_PATH):
        self.FILE_PATH = FILE_PATH
        self.min_count = 1
        self.size = 300
        self.window = 210
        self.sg = 5
        self.p2v_model = WordEmbeddingsKeyedVectors(self.size)
        self.song_p2v_model = WordEmbeddingsKeyedVectors(self.size)


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

    def get_dic(self, train, val, test, song_meta):
        song_dic = {}
        tag_dic = {}
        song_set = []
        data = train + val + test
        for q in tqdm(data):
            song_dic[str(q['id'])] = q['songs']
            tag_dic[str(q['id'])] = q['tags']
            song_set.extend(q['songs'])
        self.song_set = set(song_set)
        self.song_dic = song_dic
        self.tag_dic = tag_dic
        # total = list(map(lambda x: list(x['tags']) + [x['plylst_title']] + self._genre(x['songs']), data))
        total = list(map(lambda x: list(x['tags']) + x['plylst_title'].split() + list(map(str, x['songs'])), data))

        total = [x for x in total if len(x) > 1]
        self.total = total

    def get_w2v(self, total, min_count, size, window, sg):
        try:
            w2v_model = Word2Vec.load('./w2v_train_val_test_300_min1.model')
            self.w2v_model = w2v_model

        except:
            w2v_model = Word2Vec(total, min_count=min_count, size=size, window=window, sg=sg)
            w2v_model.save('./w2v_train_val_test_300_min1.model')
            self.w2v_model = w2v_model


    def update_p2v(self, train, val, test, w2v_model):
        ID = []
        vec = []
        for q in tqdm(train + val + test):
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

        ID = []
        vec = []
        for song_ID in tqdm(list(self.song_set)):
            try:
                ID.append(str(song_ID))
                vec.append( w2v_model.wv.get_vector(str(song_ID)))
            except KeyError:
                ID.remove(str(song_ID))
                print(song_ID)
                pass
        self.song_p2v_model.add(ID, vec)


    def get_result(self, p2v_model, song_dic, tag_dic, most_results, val):
        answers = []
        for n, q in tqdm(enumerate(val), total = len(val)):
            try:
                most_id = [x for x in p2v_model.most_similar(str(q['id']), topn=200)]
                song_id = [x[0] for x in most_id if x[1] > 0.9]
                tag_id = [x[0] for x in most_id]

                get_song = []
                get_tag = []
                if len(song_id) > 0:
                    for ID in song_id:
                        get_song += song_dic[ID]
                else:
                    get_song += q['songs']
                for ID in tag_id:
                    get_tag += tag_dic[ID]
                get_song = list(pd.value_counts(get_song)[:200].index)
                get_tag = list(pd.value_counts(get_tag)[:20].index)
                answers.append({
                    "id": q["id"],
                    "songs": remove_seen(q["songs"], get_song)[:100],
                    "tags": remove_seen(q["tags"], get_tag)[:10],
                })
            except:
                answers.append({
                  "id": most_results[n]["id"],
                  "songs": most_results[n]['songs'],
                  "tags": most_results[n]["tags"],
                })
        # check and update answer
        ss = 0
        tt = 0
        for n, q in tqdm(enumerate(answers)):
            if len(q['songs'])!=100 and len(q['songs']) > 0:

                print(q['id'])
                song_ids = []
                for song in q['songs']:
                    song_ids.extend([x[0] for x in self.song_p2v_model.most_similar(str(song), topn=150)])
                answers[n]['songs'] += remove_seen(q['songs'], song_ids)[:100-len(q['songs'])]
                ss += 1
            if len(q['songs']) !=100:
                answers[n]['songs'] += remove_seen(q['songs'], self.most_results[n]['songs'])[:100-len(q['songs'])]

            if len(q['tags'])!=10:
                answers[n]['tags'] += remove_seen(q['tags'], self.most_results[n]['tags'])[:10-len(q['tags'])]
                tt += 1

            assert len(set(answers[n]['songs'])) == 100
        print(ss, tt)
        self.answers = answers

    def run(self):
        self.get_dic(self.train, self.val, self.test, self.song_meta)
        self.get_w2v(self.total, self.min_count, self.size, self.window, self.sg)
        self.update_p2v(self.train, self.val, self.test, self.w2v_model)
        self.get_result(self.p2v_model, self.song_dic, self.tag_dic, self.most_results, self.val)
        write_json(self.answers, 'results.json')


FILE_PATH = './arena_data/'
U_space = PlaylistEmbedding(FILE_PATH)
U_space.run()