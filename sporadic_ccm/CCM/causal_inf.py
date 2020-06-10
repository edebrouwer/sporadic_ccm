from skccm.utilities import train_test_split


#Embed the time series
import skccm as ccm
import numpy as np

def causal_score(x1,x2, lag = 40, embed = 8):
    lag = lag
    embed = embed
    e1 = ccm.Embed(x1)
    e2 = ccm.Embed(x2)
    X1 = e1.embed_vectors_1d(lag,embed)
    X2 = e2.embed_vectors_1d(lag,embed)

    #split the embedded time series
    x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)

    CCM = ccm.CCM() #initiate the class

    #library lengths to test
    len_tr = len(x1tr)
    #lib_lens = np.arange(10, len_tr, len_tr/20, dtype='int')
    lib_lens = [int(len_tr)]

    #test causation
    CCM.fit(x1tr,x2tr)
    x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens)

    sc1,sc2 = CCM.score()

    return sc1, sc2
