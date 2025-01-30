
from jaxkineticmodel.parameter_estimation.training import Trainer
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load model
model_name="Smallbone2013_SerineBiosynthesis"
filepath="../../models/sbml_models/working_models/"+model_name+".xml"
model = SBMLModel(filepath)


#load data
dataset=pd.read_csv("../../datasets/Smallbone2013 - Serine biosynthesis/Smallbone2013 - Serine biosynthesis_dataset.csv",index_col=0)
#initialize the trainer object. The required inputs are model and data. We will do 300 iterations of gradient descent
trainer=Trainer(model=model,data=dataset,n_iter=300)

base_parameters=dict(zip(trainer.parameters,np.ones(len(trainer.parameters))))
parameter_sets=trainer.latinhypercube_sampling(base_parameters,
                                               lower_bound=1/10,
                                               upper_bound=10,
                                               N=5)

optimized_parameters,loss_per_iteration,global_norms=trainer.train()


#plot
fig,ax=plt.subplots(figsize=(3,3))
for i in range(5):
    ax.plot(loss_per_iteration[i])
ax.set_xlabel("Iterations")
ax.set_ylabel("Log Loss")
ax.set_yscale("log")
plt.show()


params_round1=pd.DataFrame(optimized_parameters).T
trainer.parameter_sets=params_round1
trainer.n_iter=500
optimized_parameters2,loss_per_iteration2,global_norms2=trainer.train()


#plot
fig,ax=plt.subplots(figsize=(3,3))
for i in range(5):
    plt.plot(np.concatenate((np.array(loss_per_iteration[i]),loss_per_iteration2[i])))

ax.set_xlabel("Iterations")
ax.set_ylabel("Log Loss")
ax.set_yscale("log")
plt.show()
# fig.savefig("docs/docs/images/loss_per_iter_extended.png",bbox_inches="tight")




