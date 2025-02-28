
#import
import diffrax
from jaxkineticmodel.kinetic_mechanisms import JaxKineticMechanisms as jm
from jaxkineticmodel.building_models import JaxKineticModelBuild as jkm
import jax.numpy as jnp
import jax
import equinox
import matplotlib.pyplot as plt
import pandas as pd

ReactionA = jkm.Reaction(
    name="ReactionA",
    species=['A', 'B'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Irrev_Uni(substrate="A", vmax="A_Vmax", km_substrate="A_Km"),
)

#Add reactions v1 to v3
v1 = jkm.Reaction(
    name="v1",
    species=['m1', 'm2'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Irrev_Uni(substrate="m1", vmax="A_Vmax", km_substrate="A_Km"),
)

v2 = jkm.Reaction(
    name="v2",
    species=['m2', 'm3'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Irrev_Uni(substrate="m2", vmax="B_Vmax", km_substrate="B_Km"),
)

v3 = jkm.Reaction(
    name="v3",
    species=['m2', 'm4'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Irrev_Uni(substrate="m2", vmax="C_Vmax", km_substrate="C_Km"),
)

reactions = [v1, v2, v3]
compartment_values = {'c': 1}

# initialize the kinetic model object, and then make it a simulation object through jkm.NeuralODE
kmodel = jkm.JaxKineticModelBuild(reactions, compartment_values)
kmodel_sim = jkm.NeuralODEBuild(kmodel)
print(kmodel.stoichiometric_matrix)

#define the time interval, and the initial conditions

ts = jnp.linspace(0, 10, 1000)
y0 = jnp.array([2, 0, 0, 0])
params = dict(zip(kmodel.parameter_names, jnp.array([1, 1, 1, 1, 1.5, 1])))

#jit the kmodel object. This results in a slow initial solve, but a c-compiled solve
kmodel_sim = equinox.filter_jit(kmodel_sim)
ys = kmodel_sim(ts, y0, params)
ys = pd.DataFrame(ys, columns=kmodel.species_names)

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(ts, ys['m1'], label="m1")
ax.plot(ts, ys['m2'], label="m2")
ax.plot(ts, ys['m3'], label="m3")
ax.plot(ts, ys['m4'], label="m4")
ax.set_xlabel("Time (in seconds)")
ax.set_ylabel("Concentration (in mM)")
ax.legend()
plt.show()



kmodel = jkm.JaxKineticModelBuild(reactions, compartment_values)
kmodel.add_boundary('m1', jkm.BoundaryCondition('2'))
print(kmodel.stoichiometric_matrix)

#recompile and simulate
kmodel_sim = jkm.NeuralODEBuild(kmodel)
ts = jnp.linspace(0, 10, 1000)

#we remove m1 from y0, as this is now not evaluated by solving
y0 = jnp.array([0, 0, 0])
params = dict(zip(kmodel.parameter_names, jnp.array([1, 1, 1, 1, 1.5, 1])))

#jit the kmodel object. This results in a slow initial solve, but a c-compiled solve
kmodel_sim = jax.jit(kmodel_sim)
ys = kmodel_sim(ts, y0, params)
ys = pd.DataFrame(ys, columns=kmodel.species_names)

#plot
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(ts, ys['m2'], label="m2")
ax.plot(ts, ys['m3'], label="m3")
ax.plot(ts, ys['m4'], label="m4")
ax.set_xlabel("Time (in seconds)")
ax.set_ylabel("Concentration (in mM)")
ax.legend()
plt.show()

# initialized the kinetic model object, and then make it a simulation object through jkm.NeuralODE
kmodel = jkm.JaxKineticModelBuild(reactions, compartment_values)
kmodel.add_boundary('m1', jkm.BoundaryCondition('0.5+0.3*sin(t)'))
print(kmodel.stoichiometric_matrix)

kmodel_sim = jkm.NeuralODEBuild(kmodel)
ts = jnp.linspace(0, 10, 1000)

#we remove m1 from y0, as this is now not evaluated by solving
y0 = jnp.array([0, 0, 0])
params = dict(zip(kmodel.parameter_names, jnp.array([1, 1, 1, 1, 1.5, 1])))

#jit the kmodel object. This results in a slow initial solve, but a c-compiled solve
kmodel_sim = jax.jit(kmodel_sim)
ys = kmodel_sim(ts, y0, params)
ys = pd.DataFrame(ys, columns=kmodel.species_names)

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(ts, ys['m2'], label="m2")
ax.plot(ts, ys['m3'], label="m3")
ax.plot(ts, ys['m4'], label="m4")
ax.set_xlabel("Time (in seconds)")
ax.set_ylabel("Concentration (in mM)")
ax.legend()
plt.show()

kmodel_sim = jkm.NeuralODEBuild(kmodel)
kmodel_sim._change_solver(solver=diffrax.Kvaerno3(), rtol=1e-8, atol=1e-8, icoeff=0.1)
