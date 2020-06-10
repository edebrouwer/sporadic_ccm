import numpy as np
import pandas as pd
import argparse
from .dpendulum import DPendulum 

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def df_dt(x ,dt, sigma, rho, beta):
    dx = sigma * (x[1]-x[0]) * dt
    dy = (x[0]*(rho-x[2])-x[1])*dt
    dz = (x[0]*x[1] - beta*x[2])*dt
    return np.array([dx,dy,dz])

def Lorenz(T, dt, sigma= 10, rho= 28, beta=8/3, noise_level= 0.01, couplings = [0,0]):
    N_t  = int(T//dt)
    
    x_l1 = np.zeros((N_t,3))
    x_l1[0,:] = np.array([10,15,21.1])

    x_l2 = np.zeros((N_t,3))
    x_l2[0,:] = np.array([17,12,14.2])

    x_l3 = np.zeros((N_t,3))
    x_l3[0,:] = np.array([3,8,12.4])


    for i in range(1,N_t):
        x_l1[i,:] = x_l1[i-1] + df_dt(x_l1[i-1],dt, sigma, rho, beta) + couplings[1] * np.array([x_l1[i-1,1]-x_l2[i-1,0],0,0]) * dt + noise_level*np.random.randn(3)
        x_l2[i,:] = x_l2[i-1] + df_dt(x_l2[i-1],dt, sigma, rho, beta) + couplings[0] * np.array([x_l2[i-1,1]-x_l1[i-1,0],0,0]) * dt + noise_level*np.random.randn(3)
        x_l3[i,:] = x_l3[i-1] + df_dt(x_l3[i-1],dt, sigma, rho, beta)

    return x_l1, x_l2, x_l3


def Double_pendulum(T, dt, l1, l2, m1, m2, noise_level):
    N_t = int(T//dt) #number of timepoint
    state = np.zeros((N_t,4))
    system = DPendulum(l1=l1, l2=l2, m1=m1, m2=m2, theta1 = -1)
    for i in range(N_t):
        state[i,0] = system.theta1
        state[i,1] = system.theta2
        state[i,2] = system.p1
        state[i,3] = system.p2
        system.leapfrog_step(dt)

    return state

def coupled_double_pendulum(T, dt, l_p1, l_p2, l_p3, m_p1, m_p2, m_p3, c_12, c_21,c_13, c_23, theta1, theta2, noise_level):
    N_t = int(T//dt)
    state_1 = np.zeros((N_t, 4))
    state_2 = np.zeros((N_t, 4))
    state_3 = np.zeros((N_t,4))
    systemA = DPendulum(l1=l_p1[0], l2=l_p1[1], m1=m_p1[0], m2=m_p1[1], theta1 = theta1, theta2 = theta2)
    systemB = DPendulum(l1=l_p2[0], l2=l_p2[1], m1=m_p2[0], m2=m_p2[1], theta1 = theta1, theta2 = theta2)
    systemC = DPendulum(l1=l_p3[0], l2=l_p3[1], m1=m_p3[0], m2=m_p3[1], theta1 = theta1, theta2 = theta2)
    #systemB.theta1 += 0.5
    systemA.couple(systemB,c_21)
    systemB.couple(systemA,c_12)
    systemA.couple(systemC,c_13)
    systemB.couple(systemC,c_23)
    tau = 3e-3
    for i in range(N_t):
        state_1[i,0] = systemA.theta1
        state_1[i,1] = systemA.theta2
        state_1[i,2] = systemA.p1
        state_1[i,3] = systemA.p2
        state_2[i,0] = systemB.theta1
        state_2[i,1] = systemB.theta2
        state_2[i,2] = systemB.p1
        state_2[i,3] = systemB.p2
        state_3[i,0] = systemC.theta1
        state_3[i,1] = systemC.theta2
        state_3[i,2] = systemC.p1
        state_3[i,3] = systemC.p2
        systemA.leapfrog_step(dt)
        systemB.leapfrog_step(dt)
        if ((c_13!=0) or (c_23!=0)):
            systemC.leapfrog_step(dt)
    
    return state_1, state_2, state_3




def Double_pendulum_sample(T, dt, l1,l2,m1,m2, noise_level, sample_rate, multiple_sample_rate, num_series=1, seed = 432):
    '''
    Sample from the double pendulum (theta1, theta2, p1, p2)
    The sample rate should be expressed in samples per unit of time. (on average there will be sample_rate*T sample per series) - exactly :D
    The multiple_sample_rate gives the proportion of samples wich are jointly sampled (for all dimensions)
    '''
    np.random.seed(seed)

    state = Double_pendulum(T = T, dt = dt, l1 = l1, l2 = l2, m1 = m1, m2 = m2, noise_level = noise_level)

    y = state

    N_t = int(T//dt)

    col = ["ID", "Time"] + [f"Value_{i}" for i in range(1,5)] + [f"Mask_{i}" for i in range(1,5)]

    num_samples = int(sample_rate * T)
    sample_times = np.random.choice(N_t, num_samples, replace = False)
    samples = y[sample_times,:]

    #select observations
    mask = np.ones_like(samples)
    random_mat = np.random.uniform(size = samples.shape)
    mask[random_mat>multiple_sample_rate] = 0
    samples[random_mat>multiple_sample_rate] = 0
    del random_mat

    samples = samples[mask.sum(axis=1)>0]
    sample_times = sample_times[mask.sum(axis=1)>0]
    mask    = mask[mask.sum(axis=1)>0]
    
    sample_times = sample_times*dt

    num_samples = samples.shape[0]

    if num_series > 1 :
        bins = np.linspace(0, T, num_series+1)
        id_vec = np.expand_dims(np.digitize(sample_times,bins),1)
        sample_times = sample_times - bins[id_vec-1][:,0]        
    else:
        id_vec = np.ones((num_samples,1))


    df = pd.DataFrame(np.concatenate((id_vec,np.expand_dims(sample_times,1),samples,mask),1),columns=col)
     
    df.reset_index(drop=True,inplace=True)
    return(df,y)

def Coupled_Double_pendulum_sample(T, dt, l_p1,l_p2,l_p3,m_p1,m_p2,m_p3,c_12,c_21,c_13,c_23, noise_level, sample_rate, multiple_sample_rate, theta1 = -1, theta2 = 0.5, num_series=1, seed = 432):
    '''
    Sample from the double pendulum (theta1, theta2, p1, p2)
    The sample rate should be expressed in samples per unit of time. (on average there will be sample_rate*T sample per series) - exactly :D
    The multiple_sample_rate gives the proportion of samples wich are jointly sampled (for all dimensions)
    '''
    np.random.seed(seed)

    state_1, state_2, state_3 = coupled_double_pendulum(T = T, dt = dt, l_p1 = l_p1, l_p2 = l_p2, l_p3 = l_p3, m_p1 = m_p1, m_p2 = m_p2, m_p3 = m_p3, c_12 = c_12, c_21 = c_21, c_13 = c_13, c_23 = c_23, theta1 = theta1, theta2 =theta2, noise_level = noise_level)

    y1 = state_1
    y2 = state_2
    y3 = state_3

    N_t = int(T//dt)

    col = ["ID", "Time"] + [f"Value_{i}" for i in range(1,5)] + [f"Mask_{i}" for i in range(1,5)]

    ys = [y1, y2, y3]
    dfs = []
    for i in range(len(ys)):
        
        num_samples = int(sample_rate * T)
        sample_times = np.random.choice(N_t, num_samples, replace = False)
        samples = ys[i][sample_times,:]
        

        #select observations
        mask = np.ones_like(samples)
        random_mat = np.random.uniform(size = samples.shape)
        mask[random_mat>multiple_sample_rate] = 0
        samples[random_mat>multiple_sample_rate] = 0
        del random_mat

        samples = samples[mask.sum(axis=1)>0]
        sample_times = sample_times[mask.sum(axis=1)>0]
        mask    = mask[mask.sum(axis=1)>0]
        
        sample_times = sample_times*dt

        num_samples = samples.shape[0]

        if num_series > 1 :
            bins = np.linspace(0, T, num_series+1)
            id_vec = np.expand_dims(np.digitize(sample_times,bins),1)
            sample_times = sample_times - bins[id_vec-1][:,0]        
        else:
            id_vec = np.ones((num_samples,1))


        df = pd.DataFrame(np.concatenate((id_vec,np.expand_dims(sample_times,1),samples,mask),1),columns=col)
         
        df.reset_index(drop=True,inplace=True)

        dfs += [df]
    
    return(dfs,ys)

def scaling(df,y):
    val_cols = [c for c in df.columns if "Value" in c] 
    mask_cols = [c for c in df.columns if "Mask" in c]

    for i in range(len(val_cols)):
        m = df.loc[df[mask_cols[i]]==1,val_cols[i]].mean()
        s = df.loc[df[mask_cols[i]]==1,val_cols[i]].std()
        df.loc[df[mask_cols[i]]==1,val_cols[i]] -= m
        df.loc[df[mask_cols[i]]==1,val_cols[i]] /= s
        y[:,i] = (y[:,i]-m)/s
    
    return(df,y)

if __name__=="__main__":

    T = 10000
    dt = 0.003
    l1 = 1.0
    l2 = 1.0
    m1 = 2.0
    m2 = 1.0
    noise_level = 0.
#    couplings = [0,0]
    sample_rate = 50 #10
    multiple_sample_rate = 0.6
    num_series = 1000

    df,y = Double_pendulum_sample(T = T, dt = dt, l1 = l1, l2 = l2, m1 = m1, m2 = m2, noise_level = noise_level, sample_rate = sample_rate, multiple_sample_rate = multiple_sample_rate, num_series = num_series)


    df1,y_s = scaling(df,y)
    #df2 = scaling(dfs[1])


    #Save metadata dictionary
    metadata_dict = {"T":T, "delta_t":dt, "l1": l1, "l2": l2,
                    "m1" : m1, "m2": m2, "noise_level" : noise_level,
                    "num_series" : num_series,
                    "sample_rate": sample_rate, "multiple_sample_rate": multiple_sample_rate}
    np.save(f"DPendulum_metadata.npy",metadata_dict)

    df.to_csv("DPendulum_data.csv", index = False)
    #Plot some examples and store them.
    import os
    N_examples = 10
    examples_dir = f"Dpendulum_paths_examples/"
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
    for ex in range(N_examples):
        idx = np.random.randint(low=0,high=df["ID"].nunique())
        plt.figure()
        print(idx)
        for dim in [0,1,2,3]:
            random_sample = df.loc[df["ID"]==idx].sort_values(by="Time").values
            obs_mask = random_sample[:,2+4+dim]==1
            plt.scatter(random_sample[obs_mask,1],random_sample[obs_mask,2+dim])
            plt.title("Example of generated trajectory")
            plt.xlabel("Time")
        #plt.savefig(f"{examples_dir}{args.prefix}_{ex}.pdf")
        plt.savefig(f"{examples_dir}full_example_{ex}.pdf")
        plt.close()

def compress_df(df_in,fact,time_bin = 10):
    df = df_in.copy()
    df["IDbis"] = (df.ID-1) % fact
    df.Time = df.Time + time_bin * df.IDbis
    df.ID = df.ID - df.IDbis
    df.ID = df.ID.map(dict(zip(df.ID.unique(),np.arange(df.ID.nunique()))))
    df.drop("IDbis", axis = 1,inplace = True)
    return df
