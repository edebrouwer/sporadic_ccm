import argparse
import gru_ode
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from gru_ode import data_utils
from gru_ode.Datasets.Dpendulum import datagen

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Running GRUODE on Double pendulum")
parser.add_argument('--data_name', type=str, help="data /model name to use", default="DPendulum_I_fold0")
parser.add_argument('--num_systems', type=int, help="number of systems to infer", default=2)

args = parser.parse_args()


outfile_base = "./reconstructions/"
model_name_base = args.data_name
data_name = args.data_name
num_systems = args.num_systems
variant_name = "_BEST_VAL_MSE"
#dataset = "/home/edward/Data/Causality/Dpendulum/Dpendulum_VIII_fold4_side1_data.csv"

def get_both_paths():
    
    for i in range(num_systems):
        model_name = model_name_base + f"_side{i}"
        print(f"Reconstructing time series from {model_name}")
        dataset = "./Datasets/"+data_name+f"_side{i}"+"_data.csv"
        outfile = outfile_base+data_name+f"_side{i}.csv"
        get_path(model_name,variant_name,dataset)
    
        df_recs = get_path(model_name,variant_name, dataset)
        df_recs[0].to_csv(outfile, index = False)
    return 0


def get_path(model_name, variant_name, dataset):
    device = torch.device("cuda")


    params_dict = np.load(f"trained_models/{model_name}_params.npy", allow_pickle = True).item()

    metadata = params_dict['metadata']
    params_dict = params_dict['model_params']

    N       = metadata["num_series"]
    delta_t = metadata["delta_t"]

    #c_12 = c_12
    #c_21 = c_21


    #dfs, y = datagen.Coupled_Double_pendulum_sample(T = T, dt = delta_t, l1 = metadata["l1"], l2 = metadata["l2"], m1 = metadata["m1"], m2 = metadata["m2"], c_12 = c_12, c_21 = c_21, noise_level = metadata["noise_level"], sample_rate = metadata["sample_rate"], multiple_sample_rate = metadata["multiple_sample_rate"] )


    #df1,_ = datagen.scaling(dfs[0],y[0])
    #df2,_ = datagen.scaling(dfs[1],y[1])

    #df1.ID = 0
    #df2.ID = 0

    df1 = pd.read_csv(dataset)
    df1 = datagen.compress_df(df1,df1.ID.nunique()/100,10)

    T = df1.Time.max()+0.1

    data_val_1   = gru_ode.data_utils.ODE_Dataset(panda_df = df1)
    #data_val_2   = gru_ode.data_utils.ODE_Dataset(panda_df=df2)


    dl_val_1 = DataLoader(dataset=data_val_1, collate_fn=data_utils.custom_collate_fn, shuffle=False, batch_size = 100,num_workers=1)
    #dl_val_2 = DataLoader(dataset=data_val_2, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size = 1,num_workers=1)


    model = gru_ode.NNFOwithBayesianJumps(input_size = params_dict["input_size"], hidden_size = params_dict["hidden_size"],
                                            p_hidden = params_dict["p_hidden"], prep_hidden = params_dict["prep_hidden"],
                                            logvar = params_dict["logvar"], mixing = params_dict["mixing"],
                                            full_gru_ode = params_dict["full_gru_ode"],
                                            solver = params_dict["solver"], impute = params_dict["impute"],store_hist = True)


    model.to(device)

    model.load_state_dict(torch.load(f"./trained_models/{model_name}{variant_name}.pt"))

    dl_val_list = [dl_val_1]

    dfs_rec = []
    for dl_val in dl_val_list:
        with torch.no_grad():
            for i, b in enumerate(dl_val):
                times    = b["times"]
                time_ptr = b["time_ptr"]
                X        = b["X"].to(device)
                M        = b["M"].to(device)
                obs_idx  = b["obs_idx"]
                cov      = b["cov"].to(device)

                y = b["y"]

                hT, loss, _, t_vec, p_vec, h_vec, eval_times, eval_vals = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov=cov, return_path=True)

                if params_dict["solver"] =="euler":
                    eval_vals = p_vec
                    eval_times = t_vec
                else:
                    eval_times = eval_times.cpu().numpy()

                mu, v = torch.chunk(eval_vals[:,:,:],2, dim = 2)
                mu = mu.cpu().numpy()
                v = v.cpu().numpy()

                observations = X.cpu().numpy()
               
                if params_dict["logvar"]:
                    up   = mu + np.exp(0.5*v) * 1.96
                    down = mu - np.exp(0.5*v) * 1.96
                else:
                    up   = mu + np.sqrt(v) * 1.96
                    down = mu - np.sqrt(v) * 1.96
                

                plt.figure()
                colors=["orange","green","red","blue"]
                dims = [0,1,2,3]
                plt.plot(eval_times,mu[:,0,dims[0]],"-.", c= colors[0])
                plt.plot(eval_times,mu[:,0,dims[1]],"-.", c= colors[1])
                plt.plot(eval_times,mu[:,0,dims[2]],"-.", c= colors[2])
                plt.plot(eval_times,mu[:,0,dims[3]],"-.", c= colors[3])
                #for dim in range(4):
                #    observed_idx = np.where(M.cpu().numpy()[:,dims[dim]]==1)[0]
                #    plt.scatter(times[observed_idx],observations[observed_idx,dims[dim]], c = colors[dim])
                
                #plt.scatter(eval_times,-1.2*np.ones(len(eval_times)),marker="|", label = f"DOPRI : {len(eval_times)} evals", c="green")
                plt.legend(loc = 7)
                plt.title("Prediction of trajectories for Double pendulum")
                #plt.savefig("DPendulum_debug.pdf")
                plt.close()

                break


        round_time = np.expand_dims(np.round(eval_times,3),1)
        columns = ["ID","Time"] + [f"Value_{i}" for i in range(1,mu.shape[2]+1)]

        df_rec = []
        for sim_num in range(mu.shape[1]):
            
            y_to_fill = np.concatenate((sim_num*np.ones_like(round_time),round_time,mu[:,sim_num,:]),1)
            df_rec_ = pd.DataFrame(y_to_fill, columns = columns)
            df_rec_.drop_duplicates(subset = ["Time"], keep = "last", inplace = True)
            df_rec += [df_rec_]
       
        df_rec = pd.concat(df_rec)

        dfs_rec += [df_rec]

        #plt.figure()
        #for c in columns[1:]:
        #    plt.plot(df_rec.Time.values,df_rec[c].values)
        #    plt.scatter(df1.Time.values,df1[c].values)
        #plt.savefig("reconstruction_try.pdf")
        
    return dfs_rec

if __name__ =="__main__":
    #df_recs = get_path(c_12 = 0, c_21 = 0)
    #df_recs[0].to_csv(outfile, index = False)
    get_both_paths()
