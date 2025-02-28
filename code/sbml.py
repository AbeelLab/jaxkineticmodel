
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
import jax.numpy as jnp
import pandas as pd


filepath = ("models/sbml_models/working_models/simple_sbml.xml")

# load model from file_path
model = SBMLModel(filepath)

#replaces assignment rules, boundary conditions, etc..

S=model._get_stoichiometric_matrix()

JaxKmodel = model.get_kinetic_model()

ts = jnp.linspace(0,100,2000)


#simulate given the initial conditions defined in the sbml
ys = JaxKmodel(ts=ts,
            y0=model.y0,
            params=model.parameters)
ys=pd.DataFrame(ys,columns=S.index)
