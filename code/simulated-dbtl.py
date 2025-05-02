from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
import jax.numpy as jnp
from jaxkineticmodel.utils import get_logger
from jaxkineticmodel.simulated_dbtl.dbtl import DesignBuildTestLearnCycle
logger = get_logger(__name__)

# load model (Messiha et. al (2013))
filepath = ("models/sbml_models/working_models/Messiha2013.xml")

model = SBMLModel(filepath)
# we retrieve parameters from the model


# timespan of model
ts = jnp.linspace(0, 6000, 1000)

dbtl_cycle = DesignBuildTestLearnCycle(model=model,
                                       parameters=model.parameters,
                                       initial_conditions=model.y0,
                                       timespan=ts,
                                       target=['PEP'])

# design phase
parameter_perturbations = {'lp_ADH__Kacald': [0.2, 0.5, 1, 1.5, 2],
                           'lp_ENO__kcat_ENO1': [1.2, 1.5, 1.8],
                           'lp_FBA__Kdhap': [1.1, 1.6, 1.3],
                           'lp_HXK__kcat_HXK1': [0.6, 1.1, 2, 3],
                           'lp_PGK__kcat': [1, 2, 3],
                           'lp_HXT__Vmax': [0.5, 1, 1.5],
                           'lp_GND__Kp6g_GND1': [2, 3, 4]}

dbtl_cycle.design_establish_library_elements(parameter_perturbations=parameter_perturbations)

_ = dbtl_cycle.design_assign_positions(n_positions=6)
_ = dbtl_cycle.design_assign_probabilities(probabilities_per_position=None)
strain_designs = dbtl_cycle.design_generate_strains(samples=50)

# build phase
values = dbtl_cycle.build_simulate_strains(strain_designs, plot=False)


# test phase
noised_values = dbtl_cycle.test_add_noise(values, percentage=0.1, noise_type='heteroschedastic')
data = dbtl_cycle.test_format_dataset(strain_designs=strain_designs,
                                      production_values=noised_values,
                                      reference_parameters=dbtl_cycle.parameters)

# learn phase
xgbparameters = {'tree_method': 'auto', 'reg_lambda': 1, 'max_depth': 2, "disable_default_eval_metric": 0}
alternative_params = {'num_boost_round': 10, 'early_stopping_rounds': 40}
bst, r2_scores = dbtl_cycle.learn_train_model(data=data,
                                              target="PEP",
                                              model_type="XGBoost",
                                              args=(xgbparameters, alternative_params), test_size=0.20)

dbtl_cycle.learn_validate_model(samples=50,
                                target='PEP',
                                plotting=True
                                )
