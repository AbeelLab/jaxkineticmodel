import jaxlib.xla_extension
import jax.numpy as jnp
import jax
import numpy as np
import roadrunner
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
import pandas as pd
import os
from jaxkineticmodel.utils import get_logger

logger = get_logger(__name__)

import tellurium as te  # noqa: E402

jax.config.update("jax_enable_x64", True)


def calc_euclidean(actual, predic):
    return np.sqrt(np.sum((actual - predic) ** 2))


pathname = "models/sbml_models/"
files = os.listdir(pathname)
sbml_files = []
for file in files:
    if file.endswith(".xml"):
        sbml_files.append(file)
    if file.endswith(".sbml"):
        sbml_files.append(file)

working_models_counter = 0
max_steps_reached_counter = 0
failing_models_counter = 0
discrepancy_counter = 0
for sbml_file in sbml_files:
    print(sbml_file)
    file_path = pathname + sbml_file
    try:
        model = SBMLModel(file_path)
        S = model._get_stoichiometric_matrix()
        JaxKmodel = model.get_kinetic_model()
        # simulate for jax kinetic model

        JaxKmodel = jax.jit(JaxKmodel)

        # #parameters are not yet defined

        rr = roadrunner.RoadRunner(file_path)

        rr.integrator.absolute_tolerance = 1e-10
        rr.integrator.relative_tolerance = 1e-7
        rr.integrator.initial_time_step = 1e-11
        rr.integrator.max_steps = 300000

        rr.simulate(0, 10, 200)
        sol_road_runner = rr.getSimulationData()
        ts = jnp.array(sol_road_runner["time"])


        ys = JaxKmodel(ts=ts, y0=model.y0, params=model.parameters)
        ys = pd.DataFrame(ys, columns=S.index)

        # we need to do another estimate for the error tolerance

        # calculate the MSE between two timeseries because this should be a more stable error measure
        rtols = []

        for name in S.index:
            # mse=np.sum(sol_tellurium["["+name+"]"]-ys[name])**2
            max_tell = np.max(sol_road_runner["[" + name + "]"])
            max_ys = np.max(ys[name]) + 0.0001
            max_denominator = np.max([max_tell, max_ys])
            rtol = np.abs(sol_road_runner["[" + name + "]"] - ys[name]) / max_denominator
            # cross_correlation=crosscorr(sol_tellurium["["+name+"]"],ys[name],lag=1)

            rtols.append(rtol)

        mse = np.mean(rtols)

        for i, k in enumerate(S.index):
            print(i, k)
            name = "[" + k + "]"


        S_tellurium = rr.getFullStoichiometryMatrix()
        if np.sum(np.abs(S_tellurium) - np.abs(np.array(S))) == 0:
            if mse < 0.001:
                print("numerical solve is identical: mse=" + str(mse))
                working_models_counter += 1
                os.rename(file_path, pathname + "working_models/" + sbml_file)
            else:
                print("numerical solve is not identical: mse=" + str(mse))
                discrepancy_counter += 1
                os.rename(file_path, pathname + "discrepancies/" + sbml_file)
        else:
            print("discrepancy because of S in " + sbml_file)
            discrepancy_counter += 1
            os.rename(file_path, pathname + "discrepancies/" + sbml_file)
    except jaxlib.xla_extension.XlaRuntimeError as e:
        if "maximum number of solver steps" not in str(e):
            raise
        logger.error("Maximum number of solver steps reached")
        os.rename(file_path, pathname + "max_steps_reached/" + sbml_file)
        max_steps_reached_counter += 1
    except Exception as e:
        logger.error(f"An exception of type {type(e)} was raised")
        logger.exception(e)
        os.rename(file_path, pathname + "failing_models/" + sbml_file)
        failing_models_counter += 1

print("failing_models:", failing_models_counter)
print("working_models:", working_models_counter)
print("max steps reached model:", max_steps_reached_counter)
print("discrepancies:", discrepancy_counter)
