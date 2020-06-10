import numpy as np
import pandas as pd
import gru_ode.Datasets.Dpendulum.datagen as datagen
from gru_ode.causal_inf import causal_score
import gpflow
import tensorflow as tf
from gpflow.ci_utils import ci_niter
import tqdm


num_sims = 5
model_name_base = "Dpendulum_XIII"
embed_dim = 10
#max_index_rec = 200000
#num_id_gp = 2000
comp_fact = 30

for exp in range(4,num_sims):

    print(f"Start simulation {exp}")
    data_name = f"{model_name_base}_fold{exp}"


    metadata_dict = np.load(f"/home/edward/Data/Causality/Dpendulum/{data_name}_joint_metadata.npy",allow_pickle = True).item()

    dt = metadata_dict["delta_t"]
    T = metadata_dict["T"]

    
    # Linear reconstruction causality
    df = pd.read_csv(f"/home/edward/Data/Causality/Dpendulum/{data_name}_joint_data.csv")

    df_comp = datagen.compress_df(df,fact=comp_fact).copy()

    y_gp = []
    for pendulum in [0,1,2]:
        gp_pred = []
        print(f"Inferring pendulum {pendulum}...")
        for i in range(df_comp.ID.max()+1):

            df_mvgp = df_comp.loc[df_comp.ID==i].copy()
            
            X_list = []
            Y_list = []
            for col_i in range(1,5):
                arr = df_mvgp.loc[df_mvgp[f"Mask_{pendulum*4+col_i}"]==1][["Time",f"Value_{pendulum*4+col_i}"]].values
                #print(arr.mean(axis=0))
                #print(arr.std(axis=0))
                #print(arr.shape)
                arr[:,1] -= arr[:,1].mean()
                arr[:,1] /= arr[:,1].std()

                X_list+=[np.vstack((arr[:,0],(col_i-1)*np.ones(arr.shape[0]))).T]
                Y_list+=[np.vstack((arr[:,1],(col_i-1)*np.ones(arr.shape[0]))).T]

            X_augmented = np.vstack(X_list)
            Y_augmented = np.vstack(Y_list)

            output_dim = 4  # Number of outputs
            rank = 4# Rank of W

            # Base kernel
            k = gpflow.kernels.Matern32(active_dims=[0])

            # Coregion kernel
            coreg = gpflow.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[1])

            kern = k * coreg

            lik = gpflow.likelihoods.SwitchedLikelihood(
            [gpflow.likelihoods.Gaussian() for i in range(output_dim)]
            )

            #print(i)
            #print(X_augmented.shape)

            
            # now build the GP model as normal
            m = gpflow.models.GPR((X_augmented, Y_augmented), kernel=kern)#, likelihood=lik)


            gpflow.config.set_config(gpflow.config.Config(jitter=1e-4))
            print(i)

            if (i!=32) and (i!=12) and (i!=28) and (i!=30):
            # fit the covariance function parameters
                maxiter = ci_niter(10000)
                gpflow.optimizers.Scipy().minimize(
                m.training_loss, m.trainable_variables, options=dict(maxiter=maxiter), method="L-BFGS-B",
            )
            


            Xtest = np.arange(0,int(comp_fact*10),step = dt)
            pred,_ = m.predict_f(np.vstack((Xtest,np.zeros_like(Xtest))).T)

            gp_pred +=[pred[:,0]]
        
        y_gp += [np.concatenate(gp_pred)]



    print("Done. Computing causal inference ...")
    
    x1 = y_gp[0][0::10]
    x2 = y_gp[1][0::10]
    x3 = y_gp[2][0::10]

    m_l = min(x1.shape[0],x2.shape[0],x3.shape[0])
    x1 = x1[:m_l]
    x2 = x2[:m_l]
    x3 = x3[:m_l]

    sc1_mvgp, sc2_mvgp = causal_score(x1,x2, lag = 40, embed = embed_dim)
    sc13_mvgp, sc31_mvgp = causal_score(x1,x3,lag= 40,embed = embed_dim)
    sc23_mvgp, sc32_mvgp = causal_score(x2,x3,lag= 40,embed = embed_dim)

    
    print(f"Done!")
    #print(f"Score 1 : {sc1_mvgp} - Score 2 : {sc2_mvgp}")
    
    results_entry = pd.read_csv("./results_ccm.csv")

    results_entry.loc[results_entry.dataset_name==data_name,"sc1_mvgp"] = sc1_mvgp
    results_entry.loc[results_entry.dataset_name==data_name,"sc2_mvgp"] = sc2_mvgp
    results_entry.loc[results_entry.dataset_name==data_name,"sc13_mvgp"] = sc13_mvgp
    results_entry.loc[results_entry.dataset_name==data_name,"sc31_mvgp"] = sc31_mvgp
    results_entry.loc[results_entry.dataset_name==data_name,"sc23_mvgp"] = sc23_mvgp
    results_entry.loc[results_entry.dataset_name==data_name,"sc32_mvgp"] = sc32_mvgp

    results_entry.to_csv("./results_ccm.csv",index = False)


