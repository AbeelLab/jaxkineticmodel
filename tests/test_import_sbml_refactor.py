

from jaxkineticmodel.load_sbml.sbml_load import *
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel

filepath = ("../models/sbml_models/working_models/Smallbone2013_SerineBiosynthesis.xml")

# load model from file_path
model = SBMLModel(filepath)
S=model._get_stoichiometric_matrix()
JaxKmodel = model.get_kinetic_model()

ts = jnp.linspace(0,100,2000)

#simulate given the initial conditions defined in the sbml
ys = JaxKmodel(ts=ts,
            y0=model.y0,
            params=model.parameters)
ys=pd.DataFrame(ys,columns=S.index)