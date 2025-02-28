
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel, get_global_parameters
import jax.numpy as jnp
from jaxkineticmodel.utils import get_logger
logger = get_logger(__name__)
from jaxkineticmodel.simulated_dbtl.dbtl import DesignBuildTestLearnCycle


# load model (Messiha et. al (2013))
filepath = ("models/sbml_models/working_models/Messiha2013.xml")

model = SBMLModel(filepath)
# we retrieve parameters from the model


# timespan of model
ts = jnp.linspace(0, 6000, 100)

dbtl_cycle = DesignBuildTestLearnCycle(model=model,
                                       parameters=model.parameters,
                                       initial_conditions=model.y0,
                                       timespan=ts,
                                       target=['PEP'])

# design phase
parameter_target_names = ['lp.ADH.Kacald', 'lp.ENO.kcat_ENO1',
                          'lp.FBA.Kdhap', 'lp.HXK.kcat_HXK1',
                          'lp.PGK.kcat', 'lp.HXT.Vmax', 'lp.GND.Kp6g_GND1']

parameter_perturbation_value = [[0.2, 0.5, 1, 1.5, 2],  # 'lp.ADH.Kacald'
                                [1.2, 1.5, 1.8],  # 'lp.ENO.kcat_ENO1'
                                [1.1, 1.6, 1.3],  # 'lp.FBA.Kdhap'
                                [0.6, 1.1, 2, 3],  # 'lp.HXK.kcat_HXK1'
                                [1, 2, 3],  # 'lp.PGK.kcat'
                                [0.5, 1, 1.5],  # 'lp.HXT.Vmax'
                                [2, 3, 4]]  # 'lp.GND.Kp6g_GND1'

dbtl_cycle.design_establish_library_elements(parameter_target_names,
                                             parameter_perturbation_value)

dbtl_cycle.design_assign_probabilities()

# The replacement is false means that each pro-parameter
# pair can only be chosen once from the list per strain design
strain_designs = dbtl_cycle.design_generate_strains(elements=6, samples=40, replacement=False)


# build phase
values=dbtl_cycle.build_simulate_strains(strain_designs,plot=False)

# test phase
noised_values=dbtl_cycle.test_add_noise(values,0.1,noisetype='heteroschedastic')
data=dbtl_cycle.test_format_dataset(strain_designs=strain_designs,
                               production_values=noised_values,
                               reference_parameters=dbtl_cycle.parameters)

# learn phase
xgbparameters={'tree_method': 'auto','reg_lambda':1,'max_depth':2,"disable_default_eval_metric":0}
alternative_params={'num_boost_round':10,'early_stopping_rounds':40}
bst,r2_scores=dbtl_cycle.learn_train_model(data=data,
                                           target="PEP",
                                           model_type="XGBoost",
                                           args=(xgbparameters,alternative_params),test_size=0.20)

dbtl_cycle.learn_validate_model(samples=12,
                     elements=12,
                     target='PEP',
                     plotting=True
                     )