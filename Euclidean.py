import torch 
from torch import nn, Tensor
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import random

torch.manual_seed(1)

# Setting for using only joint (Pre) or regularized with marginals (Post)
Settings = ["Pre", "Post"]

# Draws from the joint (for modeling joint)
Draws = 1000
# Number of imputations
Imp_Draws = 100
# Sample size from joint distribution
obs_xy = 20
#obs_xy = 50
# Sample size from missing patterns
SIZE = 100


mean = [0, 0]
cov = [[0.5, 0.2], [0.2, 0.5]]
np.random.seed(1)
samplesXY = np.random.multivariate_normal(mean, cov, size=obs_xy)
samplesXY1 = np.random.multivariate_normal(mean, cov, size=obs_xy)
samplesXY2 = np.random.multivariate_normal([3, 3], [[1, 0], [0, 1]], size=obs_xy)
for i in range(obs_xy):
    I = np.random.binomial(n=1, p=0.1)
    if I==0:
        samplesXY[i,:] = samplesXY1[i,:]
    elif I==1:
        samplesXY[i,:] = samplesXY2[i,:]
Y_joint = samplesXY[:,0] + 2*samplesXY[:,1] - 0.5*samplesXY[:,0]*samplesXY[:,1] + np.random.normal(0, 0.5, size=obs_xy)
samplesXY = np.insert(samplesXY, 2, Y_joint, axis=1)

df = pd.DataFrame(samplesXY)
file = "C:/Users/amval/PyWork/CausalSamps/SamplesXY.csv"
with open(file, 'w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)
    df.to_csv(file, index=False, header=True)

samplesXY = np.delete(samplesXY, 2, axis = 1)

# create multivariate normal object for later calculating normal densities
mv = multivariate_normal(mean = [0,0], cov = [[1,0], [0,1]])


samplesXY_Miss = np.random.multivariate_normal(mean, cov, size=2*SIZE)
samplesXY1_Miss = np.random.multivariate_normal(mean, cov, size=2*SIZE)
samplesXY2_Miss = np.random.multivariate_normal([3, 3], [[1, 0], [0, 1]], size=2*SIZE)
for i in range(2*SIZE):
    I = np.random.binomial(n=1, p=0.1)
    if I==0:
        samplesXY_Miss[i,:] = samplesXY1_Miss[i,:]
    elif I==1:
        samplesXY_Miss[i,:] = samplesXY2_Miss[i,:]
Y_Miss = samplesXY_Miss[:,0] + 2*samplesXY_Miss[:,1] - 0.5*samplesXY_Miss[:,0]*samplesXY_Miss[:,1] + np.random.normal(0, 0.5, size=2*SIZE)
samplesXY_Miss = np.insert(samplesXY_Miss, 2, Y_Miss, axis=1)
rand_ints1 = random.sample(range(0, 2*SIZE), SIZE)
rand_ints2 = [x for x in range(0, 2*SIZE) if x not in rand_ints1]
Y_X1Miss = Y_Miss[rand_ints1]
Y_X2Miss = Y_Miss[rand_ints2]
samplesX = samplesXY_Miss[rand_ints1, :]
samplesY = samplesXY_Miss[rand_ints2, :]
samplesX[:,1] = 0
samplesY[:,0] = 0

df = pd.DataFrame(samplesX)
file = "C:/Users/amval/PyWork/CausalSamps/SamplesX.csv"
with open(file, 'w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)
    df.to_csv(file, index=False, header=True)
df = pd.DataFrame(samplesY)
file = "C:/Users/amval/PyWork/CausalSamps/SamplesY.csv"
with open(file, 'w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)
    df.to_csv(file, index=False, header=True)

samplesX = np.delete(samplesX, 2, axis = 1)
samplesY = np.delete(samplesY, 2, axis = 1)

samplesXY = Tensor(samplesXY)
samplesX = Tensor(samplesX)
samplesY = Tensor(samplesY)
X_Train = torch.cat((torch.narrow(samplesXY, -1, 0, 1), torch.zeros(obs_xy, 1)), dim=1)
X_Train = torch.cat((X_Train, samplesX), dim=0)
Y_Train = torch.cat((torch.zeros(obs_xy, 1), torch.narrow(samplesXY, -1, 1, 1)), dim=1)
Y_Train = torch.cat((Y_Train, samplesY), dim=0)

for W in Settings:

    Pre_Post = W

    if Pre_Post == "Pre":
        # add to 1
        wx1=10/10
        wx2=0/10
        # add to 1
        wy1=10/10
        wy2=0/10
    elif Pre_Post == "Post":
        # add to 1
        wx1=5/10
        wx2=5/10
        # add to 1
        wy1=5/10
        wy2=5/10

    # For the forward flow
    zeros_for = torch.zeros(Imp_Draws, 1)
    # For the reverse flow
    zeros_rev = torch.zeros(Draws, 1)
    # For imputation (last step)
    #zeros_imp = torch.zeros(Imp_Draws, 1)
    class FlowX(nn.Module):
        def __init__(self, dim: int = 2, h: int = 333):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, h), nn.ELU(),
                nn.Linear(h, h), nn.ELU(),
                nn.Linear(h, h), nn.ELU(),
                nn.Linear(h, dim))
        
        def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
            return self.net(torch.cat((t, x_t), -1))
        
        def dstepx(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, zeros: Tensor) -> Tensor:
            t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
            x_t = torch.cat((torch.narrow(x_t, -1, 0, 1), zeros), dim=1)
            Vector = (t_end - t_start) * self(t=t_start + (t_end - t_start) / 2, x_t= x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2)
            Vector = torch.cat((torch.narrow(Vector, -1, 0, 1), zeros), dim=1)
            return Vector


    class FlowY(nn.Module):
        def __init__(self, dim: int = 2, h: int = 333):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, h), nn.ELU(),
                nn.Linear(h, h), nn.ELU(),
                nn.Linear(h, h), nn.ELU(),
                nn.Linear(h, dim))
        
        def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
            return self.net(torch.cat((t, x_t), -1))
        
        def dstepy(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, zeros: Tensor) -> Tensor:
            t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
            x_t = torch.cat((zeros, torch.narrow(x_t, -1, 1, 1)), dim=1)
            Vector = (t_end - t_start) * self(t=t_start + (t_end - t_start) / 2, x_t= x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2)
            Vector = torch.cat((zeros, torch.narrow(Vector, -1, 1, 1)), dim=1)
            return Vector


    class FlowXY(nn.Module):
        def __init__(self, dim: int = 2, h: int = 333):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, h), nn.ELU(),
                nn.Linear(h, h), nn.ELU(),
                nn.Linear(h, h), nn.ELU(),
                nn.Linear(h, dim))
        
        def forward(self, t: Tensor, xy_t: Tensor) -> Tensor:
            return self.net(torch.cat((t, xy_t), -1))
        
        def dstepxy(self, xy_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
            t_start = t_start.view(1, 1).expand(xy_t.shape[0], 1)
            
            Vector = (t_end - t_start) * self(t=t_start + (t_end - t_start) / 2, xy_t= xy_t + self(xy_t=xy_t, t=t_start) * (t_end - t_start) / 2)
            Vector[:, 0] = wx1*Vector[:, 0]
            Vector[:, 1] = wy1*Vector[:, 1]
            return Vector
        
        def step(self, xy_t: Tensor, t_start: Tensor, t_end: Tensor, zeros: Tensor) -> Tensor:
            t_start = t_start.view(1, 1).expand(xy_t.shape[0], 1)
            t_end = t_end.view(1, 1).expand(xy_t.shape[0], 1)
            return xy_t + self.dstepxy(xy_t=xy_t, t_start=t_start[i], t_end=t_end[i + 1]) + wx2*flowX.dstepx(x_t=xy_t, t_start=t_start[i], t_end=t_end[i + 1], zeros=zeros) + wy2*flowY.dstepy(x_t=xy_t, t_start=t_start[i], t_end=t_end[i + 1], zeros=zeros) 


    flowX = FlowX()
    flowY = FlowY()
    flowXY = FlowXY()


    optimizerX = torch.optim.Adam(flowX.parameters(), 1e-2)
    loss_fn_X = nn.MSELoss()
    optimizerY = torch.optim.Adam(flowY.parameters(), 1e-2)
    loss_fn_Y = nn.MSELoss()
    optimizerXY = torch.optim.Adam(flowXY.parameters(), 1e-2)
    loss_fn_XY = nn.MSELoss()


    for _ in range(10000):
        x_1 = X_Train
        x_0 = torch.randn_like(x_1)
        t = torch.rand(len(x_1), 1)
        
        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0
        
        optimizerX.zero_grad()
        loss_fn_X(flowX(t=t, x_t=x_t), dx_t).backward()
        optimizerX.step()


    for _ in range(10000):
        y_1 = Y_Train
        y_0 = torch.randn_like(y_1)
        t = torch.rand(len(y_1), 1)
        
        y_t = (1 - t) * y_0 + t * y_1
        dy_t = y_1 - y_0
        
        optimizerY.zero_grad()
        loss_fn_Y(flowY(t=t, x_t=y_t), dy_t).backward()
        optimizerY.step()


    for _ in range(10000):
        xy_1 = samplesXY
        xy_0 = torch.randn_like(xy_1)
        t = torch.rand(len(xy_1), 1)
        
        xy_t = (1 - t) * xy_0 + t * xy_1
        dxy_t = xy_1 - xy_0
        
        optimizerXY.zero_grad()
        loss_fn_XY(flowXY(t=t, xy_t=xy_t), dxy_t).backward()
        optimizerXY.step()


    xy = torch.randn(Draws, 2)

    n_steps = 1
    time_steps = torch.linspace(0, 1.0, n_steps + 1)

    for i in range(n_steps):
        xy = flowXY.step(xy_t = xy, t_start=time_steps[i], t_end=time_steps[i + 1], zeros=zeros_rev)

    # Saving the data
    df = pd.DataFrame(xy.detach())
    file = "C:/Users/amval/PyWork/CausalSamps/FullTrain_" + Pre_Post + ".csv"
    with open(file, 'w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        df.to_csv(file, index=False, header=True)



    #####################################################################################################
    ## --------------------------------- Reverse Flow (for observed X) ------------------------------- ##
    #####################################################################################################


    for k in range(0, SIZE):

        X_imp = torch.zeros(1000, 1)
        X_imp[:,0] = samplesX[k,0]
        Y_imp = torch.linspace(-5, 5, 1000)
        Y_imp = Y_imp.unsqueeze(1)
        Y_imp = torch.cat((X_imp, Y_imp), dim=1)

        xy = Y_imp

        n_steps = 1
        time_steps = torch.linspace(0, 1.0, n_steps + 1)


        # reverse time direction so that the flow is reversed
        for i in range(n_steps):
            xy = flowXY.step(xy_t = xy, t_start=time_steps[n_steps - i], t_end=time_steps[n_steps - 1 - i], zeros=zeros_rev)


        # Save probability densities at reverse flowed points
        densities = mv.pdf(xy.detach().numpy())
        # converting tensors to pandas dataframes
        densities = pd.DataFrame(densities)
        rev_data = pd.DataFrame(xy.detach())


        ## ---------------------------- Drawing Random Points From Reversed Flow ---------------------------- ##


        cdf = []
        for i in range(0, densities.shape[0]):
            cdf.append(densities[0:i].sum())
        cdf = [ x / densities[0:densities.shape[0]].sum() for x in cdf]

        # Draw from uniform, then use cdf to get random points, to be stored in rand_points
        unif = np.random.uniform(0, 1, Imp_Draws)
        rand_points = np.array([[0,0]])
        for i in range(0, Imp_Draws):
            j = 0
            cond = False
            while cond == False:
                if j == (len(cdf)-1):
                    print(cdf[j].values)
                    rand_points = np.append(rand_points, [rev_data.iloc[j,:].values], axis=0)
                    cond = True
                elif cdf[j].values < unif[i]:
                    j = j+1
                else:
                    rand_points = np.append(rand_points, [rev_data.iloc[j,:].values], axis=0)
                    cond = True
        rand_points = np.delete(rand_points, [0], axis=0)
        rand_points = torch.from_numpy(rand_points).to(torch.float32)



        ## --------------------------------- Imputing Missing Values ---------------------------------- ##


        # Plug the randomly chosen points back into the flow

        xy = rand_points

        n_steps = 1
        time_steps = torch.linspace(0, 1.0, n_steps + 1)

        # ordinary forward flow, in forward time
        for i in range(n_steps):
            xy = flowXY.step(xy_t = xy, t_start=time_steps[i], t_end=time_steps[i + 1], zeros=zeros_for)
        
        # Remove large imputed y values (shouldn't be outside of [-5, 5])
        threshold = 5
        xy = xy[torch.abs(xy[:,1]) <= threshold, :]

        # Fill in average of imputations into missing y value of the observed X
        samplesX[k,1] = torch.mean(xy.detach()[:,1])


    ## --------------------------------- Save Imputed Y Values ------------------------------- ##


    df = pd.DataFrame(samplesX)
    df["Y"] = Y_X1Miss
    file = "C:/Users/amval/PyWork/CausalSamps/SamplesX_" + Pre_Post + ".csv"
    with open(file, 'w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        df.to_csv(file, index=False, header=True)



    #####################################################################################################
    ## --------------------------------- Reverse Flow (for observed Y) ------------------------------- ##
    #####################################################################################################


    for k in range(0, SIZE):

        Y_imp = torch.zeros(1000, 1)
        Y_imp[:,0] = samplesY[k,1]
        X_imp = torch.linspace(-5, 5, 1000)
        X_imp = X_imp.unsqueeze(1)
        X_imp = torch.cat((X_imp, Y_imp), dim=1)

        xy = X_imp

        n_steps = 1
        time_steps = torch.linspace(0, 1.0, n_steps + 1)


        # reverse time direction so that the flow is reversed
        for i in range(n_steps):
            xy = flowXY.step(xy_t = xy, t_start=time_steps[n_steps - i], t_end=time_steps[n_steps - 1 - i], zeros=zeros_rev)


        # Save probability densities at reverse flowed points
        densities = mv.pdf(xy.detach().numpy())
        # converting tensors to pandas dataframes
        densities = pd.DataFrame(densities)
        rev_data = pd.DataFrame(xy.detach())


        ## ---------------------------- Drawing Random Points From Reversed Flow ---------------------------- ##


        cdf = []
        for i in range(0, densities.shape[0]):
            cdf.append(densities[0:i].sum())
        cdf = [ x / densities[0:densities.shape[0]].sum() for x in cdf]

        # Draw from uniform, then use cdf to get random points, to be stored in rand_points
        unif = np.random.uniform(0, 1, Imp_Draws)
        rand_points = np.array([[0,0]])
        for i in range(0, Imp_Draws):
            j = 0
            cond = False
            while cond == False:
                if j == (len(cdf)-1):
                    print(cdf[j].values)
                    rand_points = np.append(rand_points, [rev_data.iloc[j,:].values], axis=0)
                    cond = True
                elif cdf[j].values < unif[i]:
                    j = j+1
                else:
                    rand_points = np.append(rand_points, [rev_data.iloc[j,:].values], axis=0)
                    cond = True
        rand_points = np.delete(rand_points, [0], axis=0)
        rand_points = torch.from_numpy(rand_points).to(torch.float32)



        ## --------------------------------- Imputing Missing Values ---------------------------------- ##


        # Plug the randomly chosen points back into the (forward) flow

        xy = rand_points

        n_steps = 1
        time_steps = torch.linspace(0, 1.0, n_steps + 1)

        # ordinary forward flow, in forward time
        for i in range(n_steps):
            xy = flowXY.step(xy_t = xy, t_start=time_steps[i], t_end=time_steps[i + 1], zeros=zeros_for)
        
        # Remove large imputed x values (shouldn't be outside of [-5, 5])
        threshold = 5
        xy = xy[torch.abs(xy[:,0]) <= threshold, :]

        # Fill in average of imputations into missing X value of the observed Y
        samplesY[k,0] = torch.mean(xy.detach()[:,0])


    ## --------------------------------- Save Imputed X Values ------------------------------- ##


    df = pd.DataFrame(samplesY)
    df["Y"] = Y_X2Miss
    file = "C:/Users/amval/PyWork/CausalSamps/SamplesY_" + Pre_Post + ".csv"
    with open(file, 'w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        df.to_csv(file, index=False, header=True)


for W in Settings:

    Pre_Post = W

    if Pre_Post == "Pre":
        # add to 1
        wx1=10/10
        wx2=0/10
        # add to 1
        wy1=10/10
        wy2=0/10
    elif Pre_Post == "Post":
        # add to 1
        wx1=5/10
        wx2=5/10
        # add to 1
        wy1=5/10
        wy2=5/10


    # Bootstrapping for varibility in draws (50 bootstrapped samples)
    for l in range(0, 50):

        rand_ints_XY = [random.randint(0, obs_xy-1) for _ in range(obs_xy)]
        rand_ints_X = [random.randint(0, SIZE-1) for _ in range(SIZE)]
        rand_ints_Y = [random.randint(0, SIZE-1) for _ in range(SIZE)]

        samplesXY_cut = samplesXY[rand_ints_XY, :]
        samplesX_cut = samplesX[rand_ints_X, :]
        samplesY_cut = samplesY[rand_ints_Y, :]
        Y_X1Miss_cut = Y_X1Miss[rand_ints_X]
        Y_X2Miss_cut = Y_X2Miss[rand_ints_Y]

        df = pd.DataFrame(samplesXY_cut)
        df["Y"] = Y_joint[rand_ints_XY]
        file = "C:/Users/amval/PyWork/CausalSamps/BootstrapX1X2_" + Pre_Post + "/Boot_X1X2_" + str(l) + ".csv"
        with open(file, 'w', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)
            df.to_csv(file, index=False, header=True)

        df = pd.DataFrame(samplesX_cut)
        df["Y"] = Y_X1Miss_cut
        file = "C:/Users/amval/PyWork/CausalSamps/BootstrapX1_" + Pre_Post + "/Boot_X1_" + str(l) + ".csv"
        with open(file, 'w', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)
            df.to_csv(file, index=False, header=True)

        df = pd.DataFrame(samplesY_cut)
        df["Y"] = Y_X2Miss_cut
        file = "C:/Users/amval/PyWork/CausalSamps/BootstrapX2_" + Pre_Post + "/Boot_X2_" + str(l) + ".csv"
        with open(file, 'w', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)
            df.to_csv(file, index=False, header=True)


        flowX = FlowX()
        flowY = FlowY()
        flowXY = FlowXY()


        optimizerX = torch.optim.Adam(flowX.parameters(), 1e-2)
        loss_fn_X = nn.MSELoss()
        optimizerY = torch.optim.Adam(flowY.parameters(), 1e-2)
        loss_fn_Y = nn.MSELoss()
        optimizerXY = torch.optim.Adam(flowXY.parameters(), 1e-2)
        loss_fn_XY = nn.MSELoss()

        X_Train = torch.cat((torch.narrow(samplesXY_cut, -1, 0, 1), torch.zeros(len(rand_ints_XY), 1)), dim=1)
        X_Train = torch.cat((X_Train, samplesX_cut), dim=0)
        Y_Train = torch.cat((torch.zeros(len(rand_ints_XY), 1), torch.narrow(samplesXY_cut, -1, 1, 1)), dim=1)
        Y_Train = torch.cat((Y_Train, samplesY_cut), dim=0)
        for _ in range(10000):
            x_1 = X_Train
            x_0 = torch.randn_like(x_1)
            t = torch.rand(len(x_1), 1)
            
            x_t = (1 - t) * x_0 + t * x_1
            dx_t = x_1 - x_0
            
            optimizerX.zero_grad()
            loss_fn_X(flowX(t=t, x_t=x_t), dx_t).backward()
            optimizerX.step()


        for _ in range(10000):
            y_1 = Y_Train
            y_0 = torch.randn_like(y_1)
            t = torch.rand(len(y_1), 1)
            
            y_t = (1 - t) * y_0 + t * y_1
            dy_t = y_1 - y_0
            
            optimizerY.zero_grad()
            loss_fn_Y(flowY(t=t, x_t=y_t), dy_t).backward()
            optimizerY.step()


        for _ in range(10000):
            xy_1 = Tensor(samplesXY_cut)
            xy_0 = torch.randn_like(xy_1)
            t = torch.rand(len(xy_1), 1)
            
            xy_t = (1 - t) * xy_0 + t * xy_1
            dxy_t = xy_1 - xy_0
            
            optimizerXY.zero_grad()
            loss_fn_XY(flowXY(t=t, xy_t=xy_t), dxy_t).backward()
            optimizerXY.step()


        xy = torch.randn(Draws, 2)

        n_steps = 1
        time_steps = torch.linspace(0, 1.0, n_steps + 1)

        for i in range(n_steps):
            xy = flowXY.step(xy_t = xy, t_start=time_steps[i], t_end=time_steps[i + 1], zeros=zeros_rev)

        # Saving the data
        df = pd.DataFrame(xy.detach())
        file = "C:/Users/amval/PyWork/CausalSamps/Draws_XY_" + Pre_Post + "/Draws_" + str(l) + ".csv"
        with open(file, 'w', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)
            df.to_csv(file, index=False, header=True)

        
        #####################################################################################################
        ## --------------------------------- Reverse Flow (for observed X) ------------------------------- ##
        #####################################################################################################


        for k in range(0, samplesX_cut.size(0)):

            X_imp = torch.zeros(1000, 1)
            X_imp[:,0] = samplesX_cut[k,0]
            Y_imp = torch.linspace(-5, 5, 1000)
            Y_imp = Y_imp.unsqueeze(1)
            Y_imp = torch.cat((X_imp, Y_imp), dim=1)

            xy = Y_imp

            n_steps = 1
            time_steps = torch.linspace(0, 1.0, n_steps + 1)


            # reverse time direction so that the flow is reversed
            for i in range(n_steps):
                xy = flowXY.step(xy_t = xy, t_start=time_steps[n_steps - i], t_end=time_steps[n_steps - 1 - i], zeros=zeros_rev)


            # Save probability densities at reverse flowed points
            densities = mv.pdf(xy.detach().numpy())
            # converting tensors to pandas dataframes
            densities = pd.DataFrame(densities)
            rev_data = pd.DataFrame(xy.detach())


            ## ---------------------------- Drawing Random Points From Reversed Flow ---------------------------- ##


            cdf = []
            for i in range(0, densities.shape[0]):
                cdf.append(densities[0:i].sum())
            cdf = [ x / densities[0:densities.shape[0]].sum() for x in cdf]

            # Draw from uniform, then use cdf to get random points, to be stored in rand_points
            unif = np.random.uniform(0, 1, Imp_Draws)
            rand_points = np.array([[0,0]])
            for i in range(0, Imp_Draws):
                j = 0
                cond = False
                while cond == False:
                    if j == (len(cdf)-1):
                        print(cdf[j].values)
                        rand_points = np.append(rand_points, [rev_data.iloc[j,:].values], axis=0)
                        cond = True
                    elif cdf[j].values < unif[i]:
                        j = j+1
                    else:
                        rand_points = np.append(rand_points, [rev_data.iloc[j,:].values], axis=0)
                        cond = True
            rand_points = np.delete(rand_points, [0], axis=0)
            rand_points = torch.from_numpy(rand_points).to(torch.float32)



            ## --------------------------------- Imputing Missing Values ---------------------------------- ##


            # Plug the randomly chosen points back into the flow

            xy = rand_points

            n_steps = 1
            time_steps = torch.linspace(0, 1.0, n_steps + 1)

            # ordinary forward flow, in forward time
            for i in range(n_steps):
                xy = flowXY.step(xy_t = xy, t_start=time_steps[i], t_end=time_steps[i + 1], zeros=zeros_for)
            
            # Remove large imputed y values (shouldn't be outside of [-5, 5])
            threshold = 5
            xy = xy[torch.abs(xy[:,1]) <= threshold, :]

            # Fill in average of imputations into missing y value of the observed X
            samplesX_cut[k,1] = torch.mean(xy.detach()[:,1])


        ## --------------------------------- Save Imputed Y Values ------------------------------- ##


        df = pd.DataFrame(samplesX_cut)
        df["Y"] = Y_X1Miss_cut
        file = "C:/Users/amval/PyWork/CausalSamps/Draws_X_" + Pre_Post + "/SamplesX_" + str(l) + ".csv"
        with open(file, 'w', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)
            df.to_csv(file, index=False, header=True)



        #####################################################################################################
        ## --------------------------------- Reverse Flow (for observed Y) ------------------------------- ##
        #####################################################################################################


        for k in range(0, samplesY_cut.size(0)):

            Y_imp = torch.zeros(1000, 1)
            Y_imp[:,0] = samplesY_cut[k,1]
            X_imp = torch.linspace(-5, 5, 1000)
            X_imp = X_imp.unsqueeze(1)
            X_imp = torch.cat((X_imp, Y_imp), dim=1)

            xy = X_imp

            n_steps = 1
            time_steps = torch.linspace(0, 1.0, n_steps + 1)


            # reverse time direction so that the flow is reversed
            for i in range(n_steps):
                xy = flowXY.step(xy_t = xy, t_start=time_steps[n_steps - i], t_end=time_steps[n_steps - 1 - i], zeros=zeros_rev)


            # Save probability densities at reverse flowed points
            densities = mv.pdf(xy.detach().numpy())
            # converting tensors to pandas dataframes
            densities = pd.DataFrame(densities)
            rev_data = pd.DataFrame(xy.detach())


            ## ---------------------------- Drawing Random Points From Reversed Flow ---------------------------- ##


            cdf = []
            for i in range(0, densities.shape[0]):
                cdf.append(densities[0:i].sum())
            cdf = [ x / densities[0:densities.shape[0]].sum() for x in cdf]

            # Draw from uniform, then use cdf to get random points, to be stored in rand_points
            unif = np.random.uniform(0, 1, Imp_Draws)
            rand_points = np.array([[0,0]])
            for i in range(0, Imp_Draws):
                j = 0
                cond = False
                while cond == False:
                    if j == (len(cdf)-1):
                        print(cdf[j].values)
                        rand_points = np.append(rand_points, [rev_data.iloc[j,:].values], axis=0)
                        cond = True
                    elif cdf[j].values < unif[i]:
                        j = j+1
                    else:
                        rand_points = np.append(rand_points, [rev_data.iloc[j,:].values], axis=0)
                        cond = True
            rand_points = np.delete(rand_points, [0], axis=0)
            rand_points = torch.from_numpy(rand_points).to(torch.float32)



            ## --------------------------------- Imputing Missing Values ---------------------------------- ##


            # Plug the randomly chosen points back into the (forward) flow

            xy = rand_points

            n_steps = 1
            time_steps = torch.linspace(0, 1.0, n_steps + 1)

            # ordinary forward flow, in forward time
            for i in range(n_steps):
                xy = flowXY.step(xy_t = xy, t_start=time_steps[i], t_end=time_steps[i + 1], zeros=zeros_for)
            
            # Remove large imputed x values (shouldn't be outside of [-5, 5])
            threshold = 5
            xy = xy[torch.abs(xy[:,0]) <= threshold, :]

            # Fill in average of imputations into missing X value of the observed Y
            samplesY_cut[k,0] = torch.mean(xy.detach()[:,0])


        ## --------------------------------- Save Imputed X Values ------------------------------- ##


        df = pd.DataFrame(samplesY_cut)
        df["Y"] = Y_X2Miss_cut
        file = "C:/Users/amval/PyWork/CausalSamps/Draws_Y_" + Pre_Post + "/SamplesY_" + str(l) + ".csv"
        with open(file, 'w', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)
            df.to_csv(file, index=False, header=True)

