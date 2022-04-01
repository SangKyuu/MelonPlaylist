import pandas as pd
import json

train = pd.read_json('./train.json', typ='frame')
test = pd.read_json('./test.json', typ='frame')
val = pd.read_json('./val.json', typ='frame')
genre = pd.read_json('./genre_gn_all.json', typ='series')
song_meta = pd.read_json('./song_meta.json', typ='frame')

# tags_set = train.tags.values.tolist()
# tags_set = set([j for i in tags_set for j in i])
#
# inter_union_tags = [j for i in test.tags for j in i if j not in tags_set]
# inter_union_tags_ = [j for i in val.tags for j in i if j not in tags_set]

# song_meta.info()

SongxTag_dic = {}
for i, playlist in train.iterrows():
    for song in playlist['songs']:

        if song not in SongxTag_dic:
            SongxTag_dic[song] = []

        SongxTag_dic[song] = list(set(playlist['tags']) | set(SongxTag_dic[song]))

for i, playlist in train.iterrows():
SongxTag_dic