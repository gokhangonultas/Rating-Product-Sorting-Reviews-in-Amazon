############################Rating Product & Sorting Reviews in Amazon###############################################

###########################################Değişkenler###############################################################

#reviewerID – Kullanıcı ID’si
#Örn: A2SUAM1J3GNN3B

#asin – Ürün ID’si.
#Örn: 0000013714

#reviewerName – Kullanıcı Adı

#helpful – Faydalı yorum derecesi
#Örn: 2/3

#reviewText – Yorum
#Kullanıcının yazdığı inceleme metni

#overall – Ürün rating’i

#summary – İnceleme özeti

#unixReviewTime – İnceleme zamanı
#Unix time

#reviewTime – İnceleme zamanı
#Raw
######################################################################################################################

#Veri Hazırlanması
import numpy as np
import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df=pd.read_csv("amazon_review.csv")

df.head()

#Görev 1:Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df.head()
df.shape
df.dtypes
df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = pd.to_datetime('2021-06-19 0:0:0')
df["days"] = (current_date - df["reviewTime"]).dt.days
df["day_diff"]
df["day_diff"].quantile([0.1,0.25,0.5,0.75,0.99,1])
df["day_diff"].mean()

df.isnull().sum()
df["day_diff"] = pd.qcut(x=df["day_diff"],q=4, labels=["25","50","75","100"])
df["day_diff"] = pd.to_numeric(df["day_diff"])
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["day_diff"] == 25, "overall"].mean() * w1 / 100 + \
           dataframe.loc[ (dataframe["day_diff"] == 50), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] == 75), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] == 100), "overall"].mean() * w4 / 100


df["overall"].mean()
time_based_weighted_average(df)


time_based_weighted_average(df) - df["overall"].mean()

#Görev 2:df.head()

df.describe().T
df["comment_sort"] = df["total_vote"] - df["helpful_yes"]
df.describe().T


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],x["comment_sort"]),axis=1)

df.sort_values("wilson_lower_bound", ascending=False).head(20)[["reviewText","wilson_lower_bound"]]