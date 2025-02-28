from jaxkineticmodel.parameter_estimation.training import Trainer
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
from jaxkineticmodel.parameter_estimation.training import exponentiate_parameters
import optax
# load model
model_name = "Smallbone2013_SerineBiosynthesis"
filepath = "models/sbml_models/working_models/" + model_name + ".xml"
model = SBMLModel(filepath)

# load data
dataset = pd.read_csv("datasets/Smallbone2013 - Serine biosynthesis/Smallbone2013 - Serine biosynthesis_dataset.csv",
                      index_col=0)
#initialize the trainer object. The required inputs are model and data. We will do 300 iterations of gradient descent
trainer = Trainer(model=model, data=dataset, n_iter=300)

# latin hypercube
base_parameters = dict(zip(trainer.parameters, np.ones(len(trainer.parameters))))
parameter_sets = trainer.latinhypercube_sampling(base_parameters,
                                                 lower_bound=1 / 10,
                                                 upper_bound=10,
                                                 N=5)

optimized_parameters, loss_per_iteration, global_norms = trainer.train()

fig, ax = plt.subplots(figsize=(3, 3))
for i in range(5):
    ax.plot(loss_per_iteration[i])
ax.set_xlabel("Iterations")
ax.set_ylabel("Log Loss")
ax.set_yscale("log")
plt.show()

# next round
params_round1 = pd.DataFrame(optimized_parameters).T
trainer.parameter_sets = params_round1
trainer.n_iter = 500
optimized_parameters2, loss_per_iteration2, global_norms2 = trainer.train()

# plot
fig, ax = plt.subplots(figsize=(3, 3))
for i in range(5):
    plt.plot(np.concatenate((np.array(loss_per_iteration[i]), loss_per_iteration2[i])))

ax.set_xlabel("Iterations")
ax.set_ylabel("Log Loss")
ax.set_yscale("log")
plt.show()
# fig.savefig("docs/docs/images/loss_per_iter_extended.png",bbox_inches="tight")

# log or linear space example
trainer = Trainer(model=model, data=dataset, n_iter=300, optim_space="linear")

# optimizer change with optax optimizers
trainer = Trainer(model=model, data=dataset, n_iter=300, optimizer=optax.adam(lr=1e-3))

# own loss function
def log_mean_centered_loss_func2(params, ts, ys, model, to_include):
    """log_mean_centered_loss_func with index of state variables on which
    to train on. For example in the case of incomplete knowledge of the system"""
    params = exponentiate_parameters(params)
    mask = ~jnp.isnan(jnp.array(ys))
    ys = jnp.atleast_2d(ys)
    y0 = ys[0, :]
    y_pred = model(ts, y0, params)
    ys = jnp.where(mask, ys, 0)

    ys += 1
    y_pred += 1
    scale = jnp.mean(ys, axis=0)

    ys /= scale
    y_pred /= scale

    y_pred = jnp.where(mask, y_pred, 0)
    ys = ys[:, to_include]
    y_pred = y_pred[:, to_include]
    non_nan_count = jnp.sum(mask)
    loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
    return loss


trainer = Trainer(model=model, data=dataset, n_iter=300)
trainer._create_loss_func(log_mean_centered_loss_func2, to_include=[0])  #only include specimen 1 in the dataset
