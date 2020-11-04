import csv

import matplotlib as matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.display import Image
from sklearn.tree import export_graphviz
disease_list=[]
def return_list(disease):
    disease_list=[]
    match=disease.replace('^','_').split('_')
    ctr=1
    for group in match:
        if ctr%2==0:
            disease_list.append(group)
        ctr=ctr+1
    return disease_list
with open("dataset_uncleaned.csv") as csvfile:
    reader = csv.reader(csvfile)
    disease=""
    weight = 0
    disease_list = []
    dict_wt = {}
    dict_=defaultdict(list)
    for row in reader:

        if row[0]!="\xc2\xa0" and row[0]!="":
            disease = row[0]
            disease_list = return_list(disease)
            weight = row[1]

        if row[2]!="\xc2\xa0" and row[2]!="":
            symptom_list = return_list(row[2])

            for d in disease_list:
                for s in symptom_list:
                    dict_[d].append(s)
                dict_wt[d] = weight
with open("dataset_clean.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    for key,values in dict_.items():
        for v in values:
            #key = str.encode(key)
            key = str.encode(key).decode('utf-8')
            #.strip()
            #v = v.encode('utf-8').strip()
            #v = str.encode(v)
            writer.writerow([key,v,dict_wt[key]])
if __name__ == '__main__':
    columns = ['Source','Target','Weight']
    data = pd.read_csv("dataset_clean.csv", names=columns, encoding ="ISO-8859-1")
    data.head()
    print(data.head())
    data.to_csv("diseaseData/dataset_clean.csv",index=False)
    slist = []
    dlist = []
    with open("nodetable.csv", "w") as csvfile:
        writer = csv.writer(csvfile)

        for key, values in dict_.items():
            for v in values:
                if v not in slist:
                    writer.writerow([v, v, "symptom"])
                    slist.append(v)
            if key not in dlist:
                writer.writerow([key, key, "disease"])
                dlist.append(key)

    nt_columns = ['Id', 'Label', 'Attribute']
    nt_data = pd.read_csv("nodetable.csv", names=nt_columns, encoding="ISO-8859-1", )
    nt_data.head()
    print(nt_data.head())
    nt_data.to_csv("diseaseData/nodetable.csv", index=False)
    data = pd.read_csv("dataset_clean.csv", encoding="ISO-8859-1")
    data.head()
    print(len(data['Source'].unique()))
    print(len(data['Target'].unique()))
    df = pd.DataFrame(data)
    df_1 = pd.get_dummies(df.Target)
    df_1.head()
    print(df_1.head())
    print(df.head())
    df_s = df['Source']
    df_pivoted = pd.concat([df_s, df_1], axis=1)
    df_pivoted.drop_duplicates(keep='first', inplace=True)
    print(df_pivoted[:5])
    print(len(df_pivoted))
    cols = df_pivoted.columns
    cols = cols[1:]
    df_pivoted = df_pivoted.groupby('Source').sum()
    df_pivoted = df_pivoted.reset_index()
    print(df_pivoted[:5])
    print(len(df_pivoted))
    df_pivoted.to_csv("diseaseData/df_pivoted.csv")
    x = df_pivoted[cols]
    y = df_pivoted['Source']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    mnb = MultinomialNB()
    mnb = mnb.fit(x_train, y_train)
    print(mnb.score(x_test, y_test))
    mnb_tot = MultinomialNB()
    mnb_tot = mnb_tot.fit(x, y)
    print(mnb_tot.score(x, y))
    disease_pred = mnb_tot.predict(x)
    disease_real = y.values
    for i in range(0, len(disease_real)):
        if disease_pred[i] != disease_real[i]:
            print('Pred: {0} Actual:{1}'.format(disease_pred[i], disease_real[i]))

    print("DecisionTree")
    dt = DecisionTreeClassifier()
    clf_dt = dt.fit(x, y)
    print("Acurracy: ", clf_dt.score(x, y))
    Image(filename='tree.png')
    export_graphviz(dt,
                    out_file='diseaseData/tree.dot',
                    feature_names=cols)

    data = pd.read_csv("Training.csv")
    print(data.head())
    data = pd.read_csv("Training.csv")
    print(data.columns)
    print(len(data.columns))
    print(len(data['prognosis'].unique()))
    df = pd.DataFrame(data)
    print(df.head())
    print(len(df))
    cols = df.columns
    cols = cols[:-1]
    print(cols)
    print(len(cols))
    x = df[cols]
    y = df['prognosis']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    print("DecisionTree")
    dt = DecisionTreeClassifier()
    clf_dt = dt.fit(x_train, y_train)
    print("Acurracy: ", clf_dt.score(x_test, y_test))

    print("cross result========")
    scores = cross_val_score(dt, x_test, y_test, cv=3)
    print(scores)
    print(scores.mean())
    export_graphviz(dt,
                    out_file='diseaseData/tree.dot',
                    feature_names=cols)
    dt.__getstate__()
    importances = dt.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    features = cols
    for f in range(5):
        print("%d. feature %d - %s (%f)" % (f + 1, indices[f], features[indices[f]], importances[indices[f]]))
    export_graphviz(dt,
                    out_file='diseaseData/tree-top5.dot',
                    feature_names=cols,
                    max_depth=5
                    )
    Image(filename='tree-top5.png')
    feature_dict = {}
    for i, f in enumerate(features):
        feature_dict[f] = i
    print(feature_dict['redness_of_eyes'])
    sample_x = [i / 52 if i == 52 else i * 0 for i in range(len(features))]
    print(len(sample_x))
    sample_x = np.array(sample_x).reshape(1, len(sample_x))
    print(dt.predict(sample_x))
    print(dt.predict_proba(sample_x))

