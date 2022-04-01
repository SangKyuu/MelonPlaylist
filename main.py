import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from collections import Counter

with open('./train.json', 'r') as f:
    train = json.load(f)

with open('./test_tag_song.json', 'r', encoding='UTF-8-sig') as f:
    test_ = json.load(f)
with open('./val_tag_song.json', 'r', encoding='UTF-8-sig') as f:
    val_ = json.load(f)
with open('./test.json', 'r') as f:
    test = json.load(f)
with open('./val.json', 'r') as f:
    val = json.load(f)
# with open('./genre_gn_all.json', 'r', encoding='UTF-8-sig') as f:
#     genre = json.load(f)
# with open('./song_meta.json', 'r', encoding='UTF-8-sig') as f:
#     song_meta = json.load(f)
with open('./SongTag.json', 'r', encoding='UTF-8-sig') as f:
    SongTag = json.load(f)

def most_popular(ply, col, topk):
    c = Counter()

    for doc in ply:
        c.update(doc[col])

    top_k = c.most_common(topk)
    return c, [k for k, v in top_k]

def _song_mp_per_genre(self, song_meta, global_mp):
    res = {}

    for sid, song in song_meta.items():
        for genre in song['song_gn_gnr_basket']:
            res.setdefault(genre, []).append(sid)

    for genre, sids in res.items():
        res[genre] = Counter({k: global_mp.get(int(k), 0) for k in sids})
        res[genre] = [k for k, v in res[genre].most_common(200)]

    return res

_, song100 = most_popular(train, "songs", 100)

for i, j in zip(val, val_):
    i['tags'] = j['tags']
    i['songs'] = song100
    i['songs'] = i['songs'][:100]
    while len(i['tags']) < 10:
        i['tags'].append(len(i['tags']))
    assert len(i['tags']) == 10
    assert len(i['songs']) == 100

with open('./results.json','w') as f:
    json.dump(val, f, ensure_ascii=False)

def songs():
    for plylist in tqdm(val):
        default_songs = plylist['songs']
        plylist['weight'] = [1] * len(default_songs)

        for song in tqdm(SongTag):
            inter_num = len(set(plylist['tags']) & set(SongTag[song]))
            if inter_num > 0 and song not in plylist['songs']:
                plylist['songs'].append(song)
                plylist['weight'].append(inter_num*0.001)
        idx = np.argsort(plylist['songs'])
        songs = []
        if len(idx) >= 100:
            top100 = idx[::-1][:100]
            for i in top100:
                songs.append(plylist['songs'][i])
            plylist['songs'] = songs
        del plylist['weights']
        del songs
        print(plylist['plylst_title'])
    print(1)

def tag10():
    for ply in val:
        tag_idx = np.argsort(ply['weight'])
        tags = []
        if len(ply['tags']) >= 10:
            top10 = tag_idx[::-1][:10]
            for i in top10:
                tags.append(ply['tags'][i])
            ply['tags'] = tags


# fill tags with songs for val
def tag():
    for plylist in test:
        default_tags = plylist['tags']
        plylist['weight'] = [1]*len(default_tags)

        for song in plylist['songs']:

            if str(song) in SongTag:
                for song_tag in SongTag[str(song)]:
                    if song_tag not in plylist['tags']:
                        plylist['tags'].append(song_tag)
                        plylist['weight'].append(0.01)
                    else:
                        idx = plylist['tags'].index(song_tag)
                        plylist['weight'][idx] += 0.01
        print(plylist['plylst_title'])
    print(1)

    with open('./test_tag.json','w',  encoding='UTF-8-sig') as f:
         json.dump(test, f, ensure_ascii=False)




