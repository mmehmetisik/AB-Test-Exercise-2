
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu,pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

############################
# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. var mıdır?
############################

df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age": "mean"})

# # İki grup arasında matematiksel bir fark var gibi görünüyor. Ancak bu fark tesadüfi bir durum mu yoksa istatistiksel
# olarak anlamlı mı?

# 1. Hipotezleri kur:
# H0: M1  = M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anlamlı Farklılık Yoktur)
# H1: M1! = M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anlamlı Farklılık vardır)


# 2. Varsayımları İncele

# Normallik varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır


test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-değeri 0,05'ten küçük olduğu için H0 hipotezi reddedilir.
# Normallik varsayımı sağlanmamaktadır.

# Varyans homojenliği
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-değeri 0,05'ten büyük olduğu için H0 hipotezi reddedilemez.
# Varyanslar Homojendir.

# Normal dağılım varsayımı sağlanmadığı için non-parametrik test uygulanır.

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# p-değeri 0,05'ten küçük olduğu için H0 hipotezi reddedilir.
# Erkek ve kadın yolcuların yaş ortalamaları arasında istatistiksel olarak anlamlı bir fark vardır.