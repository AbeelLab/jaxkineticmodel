from jaxkineticmodel.load_sbml.sbml_load import *
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel

filepath = ("../../models/sbml_models/working_models/simple_sbml.xml")

# load model from file_path
model = SBMLModel(filepath)
S=model._get_stoichiometric_matrix()
JaxKmodel = model.get_kinetic_model()

ts = jnp.linspace(0,100,2000)

# parameters in sbml can be either global or local parameters.
# For gradient descent purposes we want all of them global.
params = get_global_parameters(model.model)
params = {**model.local_params, **params}

#simulate given the initial conditions defined in the sbml
ys = JaxKmodel(ts=ts,
            y0=model.y0,
            params=params)
ys=pd.DataFrame(ys,columns=S.index)