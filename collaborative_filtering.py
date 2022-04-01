from collections import Counter

import os, io
import numpy as np
import pandas as pd
import json
import distutils.dir_util

import scipy.sparse as spr
import pickle
from tqdm import tqdm


song_meta = pd.read_json("song_meta.json")
train = pd.read_json("train.json")
test = pd.read_json("val.json")

train['istrain'] = 1
test['istrain'] = 0

n_train = len(train)
n_test = len(test)

# train + test
plylst = pd.concat([train, test], ignore_index=True)

# playlist id
plylst["nid"] = range(n_train + n_test)

# id <-> nid
plylst_id_nid = dict(zip(plylst["id"],plylst["nid"]))
plylst_nid_id = dict(zip(plylst["nid"],plylst["id"]))

plylst_tag = plylst['tags']
tag_counter = Counter([tg for tgs in plylst_tag for tg in tgs])
tag_dict = {x: tag_counter[x] for x in tag_counter}

tag_id_tid = dict()
tag_tid_id = dict()
for i, t in enumerate(tag_dict):
    tag_id_tid[t] = i
    tag_tid_id[i] = t

n_tags = len(tag_dict)

plylst_song = plylst['songs']
song_counter = Counter([sg for sgs in plylst_song for sg in sgs])
song_dict = {x: song_counter[x] for x in song_counter}

song_id_sid = dict()
song_sid_id = dict()
for i, t in enumerate(song_dict):
    song_id_sid[t] = i
    song_sid_id[i] = t

n_songs = len(song_dict)

plylst['songs_id'] = plylst['songs'].map(lambda x: [song_id_sid.get(s) for s in x if song_id_sid.get(s) != None])
plylst['tags_id'] = plylst['tags'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])


plylst_use = plylst[['istrain','nid','updt_date','songs_id','tags_id']]
plylst_use.loc[:,'num_songs'] = plylst_use['songs_id'].map(len)
plylst_use.loc[:,'num_tags'] = plylst_use['tags_id'].map(len)
plylst_use = plylst_use.set_index('nid')

plylst_train = plylst_use.iloc[:n_train,:]
plylst_test = plylst_use.iloc[n_train:,:]

test = plylst_test
print(len(test))

row = np.repeat(range(n_train), plylst_train['num_songs'])
col = [song for songs in plylst_train['songs_id'] for song in songs]
dat = np.repeat(1, plylst_train['num_songs'].sum())
train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_songs))

row = np.repeat(range(n_train), plylst_train['num_tags'])
col = [tag for tags in plylst_train['tags_id'] for tag in tags]
dat = np.repeat(1, plylst_train['num_tags'].sum())
train_tags_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_tags))

train_songs_A_T = train_songs_A.T.tocsr()
train_tags_A_T = train_tags_A.T.tocsr()

def rec(pids):
    tt = 1

    res = []

    for pid in tqdm(pids):
        p = np.zeros((n_songs, 1))
        p[test.loc[pid, 'songs_id']] = 1

        val = train_songs_A.dot(p).reshape(-1)

        songs_already = test.loc[pid, "songs_id"]
        tags_already = test.loc[pid, "tags_id"]

        cand_song = train_songs_A_T.dot(val)
        cand_song_idx = cand_song.reshape(-1).argsort()[-250:][::-1]

        cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:100]
        rec_song_idx = [song_sid_id[i] for i in cand_song_idx]

        cand_tag = train_tags_A_T.dot(val)
        cand_tag_idx = cand_tag.reshape(-1).argsort()[-30:][::-1]

        cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:10]
        rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

        res.append({
        "id": plylst_nid_id[pid],
        "songs": rec_song_idx,
        "tags": rec_tag_idx
        })

        if tt % 1000 == 0:
            print(tt)

        tt += 1
    return res

answers = rec(test.index)


def write_json(data, fname):
    def _conv(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./arena_data/" + parent)
    with io.open("./arena_data/" + fname, "w", encoding="utf8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)

write_json(answers, "./results.json")
