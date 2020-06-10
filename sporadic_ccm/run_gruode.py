import argparse
import gru_ode.data_utils as data_utils
import gru_ode
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd

parser = argparse.ArgumentParser(description="Running GRUODE on Double pendulum")
parser.add_argument('--model_name', type=str, help="Model to use", default="DPendulumVIII_fold2_side1")
parser.add_argument('--dataset', type=str, help="Dataset CSV file", default="home/edward/Data/Causality/Dpendulum/Dpendulum_VIII_fold2_side1_data.csv")
parser.add_argument('--mixing', type=float, help="Mixing multiplier", default=1e-4)
parser.add_argument('--weight_decay', type=float, help="Weight decay", default=0.0005)
parser.add_argument('--seed', type=int, help="Seed for data split generation", default=432)
parser.add_argument('--solver', type=str, choices=["euler", "midpoint","dopri5"], default="euler")
parser.add_argument('--no_impute',action="store_true",default = True)
parser.add_argument('--dropout',type=float,help="dropout rate", default = 0.1)
parser.add_argument('--test_size',type=float,help="test size", default = 0.)
parser.add_argument('--rescale',type=float,help="rescaling_fact",default=1.)

args = parser.parse_args()
print(args)

model_name = args.model_name
params_dict=dict()

device  = torch.device("cuda:0")

metadata = np.load(f"./{args.dataset[:-9]}_metadata.npy",allow_pickle = True).item()


N       = metadata["num_series"]
delta_t = metadata["delta_t"]
T       = args.rescale * metadata["T"]/N

if args.test_size>0:
    train_idx = np.arange(int(N*(1-args.test_size)))
    val_idx   = np.arange(int(N*(1-args.test_size)),N)
else:
    train_idx, val_idx = train_test_split(np.arange(N),test_size=0.2, random_state=args.seed)

val_options = {"T_val": int(0.8*T), "max_val_samples": 3}
data_train = data_utils.ODE_Dataset(csv_file=args.dataset, idx=train_idx, root_dir = ".", rescale = args.rescale)
data_val   = data_utils.ODE_Dataset(csv_file=args.dataset, idx=val_idx, validation = True, val_options = val_options, root_dir = ".", rescale = args.rescale)

#Model parameters.
params_dict["input_size"]  = 4
params_dict["hidden_size"] = 50 #Hidden process
params_dict["p_hidden"]    = 10#25
params_dict["prep_hidden"] = 10#25
params_dict["logvar"]      = True
params_dict["mixing"]      = args.mixing
params_dict["delta_t"]     = delta_t
params_dict["dataset"]     = args.dataset
params_dict["jitter"]      = 0
params_dict["gru_bayes"]   = "masked_mlp" #!!
params_dict["full_gru_ode"] = True
params_dict["solver"]      = args.solver
params_dict["impute"]      = not args.no_impute
params_dict["dropout"]     = args.dropout

params_dict["T"]           = T

#Model parameters and the metadata of the dataset used to train the model are stored as a single dictionnary.
summary_dict ={"model_params":params_dict,"metadata":metadata}
np.save(f"trained_models/{model_name}_params.npy",summary_dict)

dl     = DataLoader(dataset=data_train, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size=100,num_workers=4)
dl_val = DataLoader(dataset=data_val, collate_fn=data_utils.custom_collate_fn, shuffle=False, batch_size = len(data_val),num_workers=1)

## the neural negative feedback with observation jumps
model = gru_ode.NNFOwithBayesianJumps(input_size = params_dict["input_size"], hidden_size = params_dict["hidden_size"],
                                        p_hidden = params_dict["p_hidden"], prep_hidden = params_dict["prep_hidden"],
                                        logvar = params_dict["logvar"], mixing = params_dict["mixing"],
                                        full_gru_ode = params_dict["full_gru_ode"],dropout_rate = params_dict["dropout"],
                                        solver = params_dict["solver"], impute = params_dict["impute"])
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
epoch_max = 2

params_dict=dict()

#Training
mse_val_best = 1000
for epoch in range(epoch_max):
    model.train()
    optimizer.zero_grad()
    mse_tr  = 0.0
    loss_tr = 0.0
    num_tr  = 0.0
    for i, b in tqdm.tqdm(enumerate(dl)):

        times    = b["times"]
        time_ptr = b["time_ptr"]
        X        = b["X"].to(device) # + torch.randn_like(b["X"], device = device) * 1e-1 #Noise injection test
        M        = b["M"].to(device)
        obs_idx  = b["obs_idx"]
        cov      = b["cov"].to(device)

        y = b["y"]


        hT, loss, mse, _, t_vec, p_vec, h_vec, eval_times,_ = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov=cov, return_path=True, return_mse=True)
        loss.backward()
        if i%1==0: #10
            optimizer.step()
            optimizer.zero_grad()
#        import ipdb; ipdb.set_trace()        
#        t_vec = np.around(t_vec,str(delta_t)[::-1].find('.')).astype(np.float32) #Round floating points error in the time vector.
#        p_val     = data_utils.extract_from_path(t_vec,p_vec,times_val,times_idx)
#        m, v      = torch.chunk(p_val,2,dim=1)
#        last_loss = (data_utils.log_lik_gaussian(X_val,m,v)*M_val).sum()
#        mse_loss  = (torch.pow(X_val - m, 2) * M_val).sum()

 #       loss_tr  += last_loss.cpu().numpy()
        mse_tr   += mse.cpu().detach().numpy()
        num_tr   += M.sum().cpu().detach().numpy()

 #   loss_tr /= num_tr
    mse_tr  /= num_tr
    with torch.no_grad():
        mse_val  = 0
        loss_val = 0
        num_obs  = 0
        model.eval()
        for i, b in enumerate(dl_val):
            times    = b["times"]
            time_ptr = b["time_ptr"]
            X        = b["X"].to(device)
            M        = b["M"].to(device)
            obs_idx  = b["obs_idx"]
            cov      = b["cov"].to(device)

            X_val     = b["X_val"].to(device)
            M_val     = b["M_val"].to(device)
            times_val = b["times_val"]
            times_idx = b["index_val"]

            y = b["y"]

            hT, loss, _, t_vec, p_vec, h_vec, eval_times,_ = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov=cov, return_path=True)
            t_vec = np.around(t_vec,str(delta_t)[::-1].find('.')).astype(np.float32) #Round floating points error in the time vector.

            p_val     = data_utils.extract_from_path(t_vec,p_vec,times_val,times_idx)
            m, v      = torch.chunk(p_val,2,dim=1)
            last_loss = (data_utils.log_lik_gaussian(X_val,m,v)*M_val).sum()
            mse_loss  = (torch.pow(X_val - m, 2) * M_val).sum()

            loss_val += last_loss.cpu().numpy()
            mse_val  += mse_loss.cpu().numpy()
            num_obs  += M_val.sum().cpu().numpy()

        loss_val /= num_obs
        mse_val  /= num_obs
        print(f"Mean validation loss at epoch {epoch}: nll={loss_val:.5f}, mse={mse_val:.5f}  (num_obs={num_obs}) (TRAIN: nll={loss_tr:.5f}, mse={mse_tr:.5}, num_tr={num_tr})")
        if epoch % 20 == 0:
            model_file = f"./trained_models/{model_name}_{epoch}.pt"
            torch.save(model.state_dict(),model_file)
        if mse_val < mse_val_best:
            print("New best validation MSE")
            torch.save(model.state_dict(),f"./trained_models/{model_name}_BEST_VAL_MSE.pt")
            mse_val_best = mse_val

print(f"Last validation log likelihood : {loss_val}")
print(f"Last validation MSE : {mse_val}")
df_file_name = "./trained_models/DPendulum_results.csv"
df_res = pd.DataFrame({"Name" : [model_name], "LogLik" : [loss_val], "MSE" : [mse_val], "Dataset": [args.dataset], "Seed": [args.seed]})
if os.path.isfile(df_file_name):
    df = pd.read_csv(df_file_name)
    df = df.append(df_res)
    df.to_csv(df_file_name,index=False)
else:
    df_res.to_csv(df_file_name,index=False)


model_file = f"./trained_models/{model_name}.pt"
torch.save(model.state_dict(),model_file)
print(f"Saved model into '{model_file}'.")
