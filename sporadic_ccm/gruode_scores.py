
import numpy as np
import pandas as pd

from gru_ode.causal_inf import causal_score

samples_per_sec = 100
time_bins = 10 #seconds
embed_dim = 8

for fold in range(5):
    print(f"Computing fold : {fold} ...")
    data_name = f"Dpendulum_XIII_fold{fold}"
    reconstruction_name = f"XIII_fold{fold}"

    #Time Series1
    df_o = pd.read_csv(f"/home/edward/Data/Causality/Dpendulum/{data_name}_side0_data.csv")
    df_r = pd.read_csv(f"./reconstructions/{reconstruction_name}_side0.csv")
    y = np.load(f"/home/edward/Data/Causality/Dpendulum/{data_name}_side0_full.npy")

    n_chunks = df_o.ID.nunique()/100

    df_r.Time = df_r.Time + df_r.ID*(n_chunks*samples_per_sec)
    df_ode0 = df_r.copy()

    #Time Series2
    df_o = pd.read_csv(f"/home/edward/Data/Causality/Dpendulum/{data_name}_side1_data.csv")
    df_r = pd.read_csv(f"./reconstructions/{reconstruction_name}_side1.csv")
    y = np.load(f"/home/edward/Data/Causality/Dpendulum/{data_name}_side1_full.npy")

    n_chunks = df_o.ID.nunique()/100

    df_r.Time = df_r.Time + df_r.ID*(n_chunks*samples_per_sec)
    df_ode1 = df_r.copy()

    #Time Series2
    df_o = pd.read_csv(f"/home/edward/Data/Causality/Dpendulum/{data_name}_side2_data.csv")
    df_r = pd.read_csv(f"./reconstructions/{reconstruction_name}_side2.csv")
    y = np.load(f"/home/edward/Data/Causality/Dpendulum/{data_name}_side2_full.npy")

    n_chunks = df_o.ID.nunique()/100

    df_r.Time = df_r.Time + df_r.ID*(n_chunks*samples_per_sec)
    df_ode2 = df_r.copy()

    x1 = df_ode0.Value_1.values[0::10]
    x2 = df_ode1.Value_1.values[0::10]
    x3 = df_ode2.Value_1.values[0::10]

    m_l = min(x1.shape[0],x2.shape[0],x3.shape[0])
    x1 = x1[:m_l]
    x2 = x2[:m_l]
    x3 = x3[:m_l]

    sc1_gruode, sc2_gruode = causal_score(x1,x2, lag = 40, embed = embed_dim)
    sc13_gruode, sc31_gruode = causal_score(x1,x3,lag=500,embed = embed_dim)
    sc23_gruode, sc32_gruode = causal_score(x2,x3,lag=40,embed = embed_dim)

    print(sc31_gruode)

    results_entry = pd.read_csv("./results_ccm.csv")

    results_entry.loc[results_entry.dataset_name==data_name,"sc1_gru_ode"] = sc1_gruode
    results_entry.loc[results_entry.dataset_name==data_name,"sc2_gru_ode"] = sc2_gruode
    results_entry.loc[results_entry.dataset_name==data_name,"sc13_gru_ode"] = sc13_gruode
    results_entry.loc[results_entry.dataset_name==data_name,"sc31_gru_ode"] = sc31_gruode
    results_entry.loc[results_entry.dataset_name==data_name,"sc23_gru_ode"] = sc23_gruode
    results_entry.loc[results_entry.dataset_name==data_name,"sc32_gru_ode"] = sc32_gruode

    results_entry.to_csv("./results_ccm.csv",index = False)



def compress_df(df,fact,time_bin = 10):
    df["IDbis"] = (df.ID-1) % fact
    df.Time = df.Time + time_bin * df.IDbis
    df.ID = df.ID - df.IDbis
    df.ID = df.ID.map(dict(zip(df.ID.unique(),np.arange(df.ID.nunique()))))
    df.drop("IDbis", axis = 1,inplace = True)
    return df
