"""Implementation of the glycolysis model from
Lao-Martil, D., Schmitz, J. P., Teusink, B., & van Riel, N. A. (2023).
Elucidating yeast glycolytic dynamics at steady state growth and glucose pulses
through kinetic metabolic modeling. Metabolic Engineering, 77, 128-142. Stoichiometry is
changed according to the paper."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import sympy

from jaxkineticmodel.building_models import JaxKineticModelBuild as jkm
from jaxkineticmodel.kinetic_mechanisms import JaxKineticMechanisms as jm
from jaxkineticmodel.kinetic_mechanisms import JaxKineticMechanismsCustom as jcm
from jaxkineticmodel.kinetic_mechanisms import JaxKineticModifiers as modifier
from jaxkineticmodel.load_sbml.export_sbml import SBMLExporter

v_glt = jkm.Reaction(
    name="v_GLT",
    species=['ECglucose', 'ICglucose'],
    stoichiometry=[-1, 1],
    compartments=['e', 'c'],
    mechanism=jm.Jax_Facilitated_Diffusion(substrate="ECglucose", product="ICglucose",
                                           vmax="p_GLT_VmGLT", km_internal="p_GLT_KmGLTGLCi",
                                           km_external="p_GLT_KmGLTGLCo", ))

v_hxk = jkm.Reaction(
    name="v_HXK",
    species=['ICATP', 'ICglucose', "ICADP", 'ICG6P', "ICT6P"],
    stoichiometry=[-1, -1, 1, 1, 0],
    compartments=['c', 'c', 'c', 'c'],
    mechanism=jcm.Jax_MM_Rev_BiBi_w_Inhibition(substrate1="ICATP", substrate2="ICglucose",
                                               product1="ICADP", product2="ICATP", modifier="ICT6P",
                                               vmax="p_HXK_Vmax", k_equilibrium="p_HXK1_Keq",
                                               km_substrate1="p_HXK1_Katp",
                                               km_substrate2="p_HXK1_Kglc", km_product1="p_HXK1_Kadp",
                                               km_product2="p_HXK1_Kg6p", ki_inhibitor="p_HXK1_Kt6p", ))
v_nth1 = jkm.Reaction(
    name="v_NTH1",
    species=['ICtreh', 'ICglucose'],
    stoichiometry=[-1, 2],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Irrev_Uni(substrate="ICtreh",
                                  vmax="p_NTH1_Vmax",
                                  km_substrate="p_NTH1_Ktre"))

v_pgi = jkm.Reaction(
    name="v_PGI",
    species=['ICG6P', 'ICF6P'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Rev_UniUni(substrate="ICG6P", product="ICF6P",
                                   vmax="p_PGI1_Vmax", k_equilibrium="p_PGI1_Keq",
                                   km_substrate="p_PGI1_Kg6p", km_product="p_PGI1_Kf6p", ))

v_sinkg6p = jkm.Reaction(
    name="v_sinkG6P",
    species=['ICG6P', 'ICPHOS'],
    stoichiometry=[-1, 1],
    compartments=['c'],
    mechanism=jm.Jax_MM_Sink(substrate="ICG6P",
                             v_sink="poly_sinkG6P",
                             km_sink="km_sinkG6P"))

v_sinkf6p = jkm.Reaction(name="v_sinkf6P",
                         species=['ICF6P', 'ICPHOS'],
                         stoichiometry=[1, -1],
                         compartments=['c'],
                         mechanism=jm.Jax_MM_Sink(substrate="ICF6P",
                                                  v_sink="poly_sinkF6P",
                                                  km_sink="km_sinkF6P"))

v_pgm1 = jkm.Reaction(
    name="v_PGM1",
    species=['ICG1P', 'ICG6P'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Rev_UniUni(substrate="ICG1P", product="ICG6P",
                                   vmax="p_PGM1_Vmax", k_equilibrium="p_PGM1_Keq",
                                   km_substrate="p_PGM1_Kg1p",
                                   km_product="p_PGM1_Kg6p"))  # to do v_TPS1 for 2nd rate law

## look into how to deal with arbitrary arguments in compute()
mech_tps1 = jm.Jax_MM_Irrev_Bi(
    substrate1="ICG6P",
    substrate2="ICG1P",
    vmax="p_TPS1_Vmax",
    km_substrate1="p_TPS1_Kg6p",
    km_substrate2="p_TPS1_Kudp_glc",
)
mech_tps1.add_modifier(modifier.SimpleActivator(activator="ICF6P", k_A="p_TPS1_KmF6P"))

#modifiers need to be added with 0 stoichiometry
v_tps1 = jkm.Reaction(
    name="v_TPS1",
    species=['ICG6P', 'ICG1P', 'ICATP', 'ICT6P', 'ICADP', 'ICPHOS', 'ICF6P'],
    stoichiometry=[-1, -1, -1, 1, 1, 2, 0],
    compartments=['c', 'c', 'c', 'c', 'c'],
    mechanism=mech_tps1)


v_tps2 = jkm.Reaction(
    name="v_TPS2",
    species=["ICT6P", "ICtreh", "ICPHOS"],
    stoichiometry=[-1, 1, 1],
    compartments=['c', 'c', 'c'],
    mechanism=jcm.Jax_MM_Irrev_Bi_w_Inhibition(substrate="ICT6P",
                                               product="ICPHOS",
                                               vmax="p_TPS2_Vmax",
                                               km_substrate1="p_TPS2_Kt6p",
                                               ki="p_TPS2_Kpi"))

v_pfk = jkm.Reaction(
    name="v_PFK",
    species=['ICF6P', 'ICATP', 'ICFBP', 'ICADP', 'ICAMP'],
    stoichiometry=[-1, -1, 1, 1, 0],
    compartments=['c', 'c', 'c', 'c'],
    mechanism=jcm.Jax_PFK(substrate1="ICF6P", substrate2="ICATP", product="ICFBP",
                          modifiers="ICAMP", vmax="p_PFK_Vmax", kr_F6P="p_PFK_Kf6p",
                          kr_ATP="p_PFK_Katp", gr="p_PFK_gR", c_ATP="p_PFK_Catp",
                          ci_ATP="p_PFK_Ciatp", ci_AMP="p_PFK_Camp", ci_F26BP="p_PFK_Cf26bp",
                          ci_F16BP="p_PFK_Cf16bp", l="p_PFK_L", kATP="p_PFK_Kiatp", kAMP="p_PFK_Kamp",
                          F26BP="p_PFK_F26BP", kF26BP="p_PFK_Kf26bp", kF16BP="p_PFK_Kf16bp"))
v_ald = jkm.Reaction(
    name="v_ALD",
    species=['ICFBP', 'ICGAP', 'ICDHAP'],
    stoichiometry=[-1, 1, 1],
    compartments=['c', 'c', 'c'],
    mechanism=jm.Jax_MM_Rev_UniBi(substrate="ICFBP", product1="ICGAP", product2="ICDHAP",
                                  vmax="p_FBA1_Vmax", k_equilibrium="p_FBA1_Keq",
                                  km_substrate="p_FBA1_Kf16bp", km_product1="p_FBA1_Kglyceral3p",
                                  km_product2="p_FBA1_Kdhap", ))

v_tpi1 = jkm.Reaction(
    name="v_TPI1",
    species=['ICDHAP', 'ICGAP'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Rev_UniUni(substrate="ICDHAP", product="ICGAP", vmax="p_TPI1_Vmax",
                                   k_equilibrium="p_TPI1_Keq", km_substrate="p_TPI1_Kdhap",
                                   km_product="p_TPI1_Kglyceral3p", ))

v_sinkgap = jkm.Reaction(
    name="v_sinkGAP",
    species=['ICGAP', 'ICPHOS'],
    stoichiometry=[1, -1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Sink(substrate="ICGAP", v_sink="poly_sinkGAP",
                             km_sink="poly_sinkGAP"))

## look into how to deal with arbitrary arguments in compute()
v_g3pdh = jkm.Reaction(
    name="v_3GPDH",
    species=['ICDHAP', 'ICNADH', 'ICG3P', 'ICNAD', 'ICFBP', 'ICATP', 'ICADP'],
    stoichiometry=[-1, -1, 1, 1, 0, 0, 0],
    compartments=['c', 'c', 'c', 'c'],
    mechanism=jcm.G3PDH_Func_TEMP(substrate1="ICDHAP",
                                  substrate2="ICNADH",
                                  product1="ICG3P",
                                  product2="ICNAD",
                                  modifier1="ICFBP",
                                  modifier2="ICATP",
                                  modifier3="ICADP",
                                  vmax="p_GPD1_Vmax",
                                  k_equilibrium="p_GPD1_Keq",
                                  km_substrate1="p_GPD1_Kdhap",
                                  km_substrate2="p_GPD1_Knadh",
                                  km_product1="p_GPD1_Kglyc3p",
                                  km_product2="p_GPD1_Knad",
                                  ka1="p_GPD1_Kf16bp",
                                  ka2="p_GPD1_Katp",
                                  ka3="p_GPD1_Kadp", ))

v_gapdh = jkm.Reaction(
    name="v_GAPDH",
    species=['ICGAP', 'ICNAD', 'ICPHOS', 'ICBPG', 'ICNADH'],
    stoichiometry=[-1, -1, -1, 1, 1],
    compartments=['c', 'c', 'c', 'c', 'c'],
    mechanism=jcm.Jax_MM_Ordered_Bi_Tri(substrate1="ICGAP", substrate2="ICNAD",
                                        substrate3="ICPHOS", product1="ICBPG",
                                        product2="ICNADH",
                                        vmax="p_GAPDH_Vmax", k_equilibrium="p_TDH1_Keq",
                                        km_substrate1="p_TDH1_Kglyceral3p", km_substrate2="p_TDH1_Knad",
                                        ki="p_TDH1_Kpi", km_product1="p_TDH1_Kglycerate13bp",
                                        km_product2="p_TDH1_Knadh"))

v_pgk = jkm.Reaction(
    name="v_PGK",
    species=['ICBPG', 'ICADP', 'IC3PG', 'ICATP'],
    stoichiometry=[-1, -1, 1, 1],
    compartments=['c', 'c', 'c', 'c'],
    mechanism=jm.Jax_MM_Rev_BiBi(substrate1="ICBPG",
                                 substrate2="ICADP",
                                 product1="IC3PG",
                                 product2="ICATP",
                                 vmax="p_PGK_VmPGK",
                                 k_equilibrium="p_PGK_KeqPGK",
                                 km_substrate1="p_PGK_KmPGKBPG",
                                 km_substrate2="p_PGK_KmPGKADP",
                                 km_product1="p_PGK_KmPGKP3G",
                                 km_product2="p_PGK_KmPGKATP", ))

v_sink3pga = jkm.Reaction(
    name="v_sink3PGA",
    species=['IC3PG', 'ICPHOS'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Sink(substrate="IC3PG", v_sink="poly_sinkP3G", km_sink="km_sinkP3G"))

mech_hor2 = jm.Jax_MM_Irrev_Uni(substrate="ICG3P",
                                vmax="p_HOR2_Vmax",
                                km_substrate="p_HOR2_Kglyc3p")

mech_hor2.add_modifier(modifier.SimpleInhibitor(inhibitor="ICPHOS", k_I="p_HOR2_Kpi"))

## look into how to deal with arbitrary arguments in compute()
v_hor2 = jkm.Reaction(name="v_HOR2",
                      species=['ICG3P', 'ICglyc', 'ICPHOS'],
                      stoichiometry=[-1, 1, 1],
                      compartments=['c', 'c', 'c'],
                      mechanism=mech_hor2)

### next up  v_GlycT

mech_glyct_ic = jm.Jax_Diffusion(substrate="ICglyc",
                                 enzyme="f_GLYCEROL_e",
                                 transport_coef="p_GlycerolTransport")
v_glyct_ic = jkm.Reaction(
    name="v_GlycT_IC",
    species=['ICglyc'],
    stoichiometry=[-1],
    compartments=['c'],
    mechanism=mech_glyct_ic)

mech_glyct_ec = jm.Jax_Diffusion(substrate="ICglyc",
                                 enzyme="f_GLYCEROL_e",
                                 transport_coef="p_GlycerolTransport")
mech_glyct_ec.add_modifier(modifier.BiomassModifier(biomass="ECbiomass"))

v_glyct_ec = jkm.Reaction(
    name="v_GlycT_EC",
    species=['ICglyc', 'ECglyc'],
    stoichiometry=[0, 1],
    compartments=['c', 'e'],
    mechanism=mech_glyct_ec)

v_pgm = jkm.Reaction(
    name="v_PGM",
    species=['IC3PG', 'IC2PG'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Rev_UniUni(substrate="IC3PG",
                                   product="IC2PG",
                                   vmax="p_PGM_Vm",
                                   k_equilibrium="p_PGM_Keq",
                                   km_substrate="p_PGM_K3pg",
                                   km_product="p_PGM_K2pg", ))

v_eno2 = jkm.Reaction(
    name="v_ENO2",
    species=['IC2PG', 'ICPEP'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Rev_UniUni(substrate="IC2PG",
                                   product="ICPEP",
                                   vmax="p_ENO1_Vm",
                                   k_equilibrium="p_ENO1_Keq",
                                   km_substrate="p_ENO1_K2pg",
                                   km_product="p_ENO1_Kpep", ))

v_sinkpep = jkm.Reaction(
    name="v_sinkPEP",
    species=['ICPEP', 'ICPHOS'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Sink(substrate="ICPEP",
                             v_sink="poly_sinkPEP",
                             km_sink="km_sinkPEP"))

v_pyk1 = jkm.Reaction(
    name='v_PYK1',
    species=['ICPEP', 'ICADP', 'ICPYR', 'ICATP', 'ICFBP'],
    stoichiometry=[-1, -1, 1, 1, 0],
    compartments=['c', 'c', 'c', 'c'],
    mechanism=jcm.Jax_Hill_Irreversible_Bi_Activation(substrate1="ICPEP",
                                                      substrate2="ICADP",
                                                      activator="ICFBP",
                                                      product="ICATP",
                                                      vmax="p_PYK1_Vm",
                                                      hill="p_PYK1_hill",
                                                      k_substrate1="p_PYK1_Kpep",
                                                      k_substrate2="p_PYK1_Kadp",
                                                      k_product="p_PYK1_Katp",
                                                      k_activator="p_PYK1_Kf16bp",
                                                      l="p_PYK1_L", ))
v_sinkpyr = jkm.Reaction(
    name='v_sinkPYR',
    species=['ICPYR'],
    stoichiometry=[-1],
    compartments=['c'],
    mechanism=jm.Jax_MM_Sink(substrate="ICPYR",
                             v_sink="poly_sinkPYR",
                             km_sink="km_sinkPYR"))

v_adh = jkm.Reaction(
    name='v_ADH',
    species=['ICNAD', 'ICETOH', 'ICNADH', 'ICACE'],
    stoichiometry=[1, 1, -1, -1],
    compartments=['c', 'c', 'c', 'c'],
    mechanism=jcm.Jax_ADH(NAD="ICNAD",
                          ETOH="ICETOH",
                          NADH="ICNADH",
                          ACE="ICACE",
                          vmax="p_ADH_VmADH",
                          k_equilibrium="p_ADH_KeqADH",
                          km_substrate1="p_ADH_KiADHNAD",
                          km_substrate2="p_ADH_KmADHETOH",
                          km_product1="p_ADH_KmADHACE",
                          km_product2="p_ADH_KmADHNADH",
                          ki_substrate1="p_ADH_KiADHNAD",
                          ki_substrate2="p_ADH_KiADHETOH",
                          ki_product1="p_ADH_KiADHACE",
                          ki_product2="p_ADH_KiADHNADH",
                          exprs_cor="p_ADH_ExprsCor", ))

v_sinkace = jkm.Reaction(
    name='v_sinkACE',
    species=['ICACE'],
    stoichiometry=[-1],
    compartments=['c'],
    mechanism=jm.Jax_MM_Sink(substrate="ICACE",
                             v_sink="poly_sinkACE",
                             km_sink="km_sinkACE"))

#from a modelling perspective this is awkward. we have a rate that only needs to be modified for ECETOH, not ICETOH.
# where therefore need to split the stoichiometry as well, while maintaining the same parameters
mech_etoht_ic = jm.Jax_Diffusion(substrate="ICETOH",
                                 enzyme="f_ETOH_e",
                                 transport_coef="p_kETOHtransport")




#only outgoing
v_etoht_ic = jkm.Reaction(
    name="v_ETOHT_IC",
    species=['ICETOH'],
    stoichiometry=[-1, 0],
    compartments=['c'],
    mechanism=mech_etoht_ic)

mech_etoht_ec = jm.Jax_Diffusion(substrate="ICETOH",
                                 enzyme="f_ETOH_e",
                                 transport_coef="p_kETOHtransport")
mech_etoht_ec.add_modifier(modifier.BiomassModifier(biomass='ECbiomass'))


v_etoht_ec = jkm.Reaction(
    name="v_ETOHT_EC",
    species=['ICETOH', 'ECETOH'],
    stoichiometry=[0, 1],
    compartments=['c', 'e'],
    mechanism=mech_etoht_ec)



v_atpmito = jkm.Reaction(
    name='v_ATPMITO',
    species=['ICATP', 'ICPHOS', 'ICADP'],
    stoichiometry=[1, -1, -1],
    compartments=['c', 'c', 'c'],
    mechanism=jm.Jax_MM_Irrev_Bi(substrate1="ICADP",
                                 substrate2="ICPHOS",
                                 vmax="p_mitoVmax",
                                 km_substrate1="p_mitoADPKm",
                                 km_substrate2="p_mitoPiKm"))

v_atpase = jkm.Reaction(
    name="v_ATPASE",
    species=['ICATP', 'ICPHOS', 'ICADP'],
    stoichiometry=[-1, 1, 1],
    compartments=['c', 'c', 'c'],
    mechanism=jcm.Jax_ATPase(substrate="ICATP",
                             product="ICADP",
                             ATPase_ratio="p_ATPase_ratio"))
v_adk = jkm.Reaction(
    name="v_ADK",
    species=['ICADP', 'ICATP', 'ICAMP'],
    stoichiometry=[-2, 1, 1],
    compartments=['c', 'c', 'c'],
    mechanism=jcm.Jax_MA_Rev_Bi(substrate1="ICADP",
                                substrate2="ICADP",
                                product1="ICATP",
                                product2="ICAMP",
                                k_equilibrium="p_ADK1_Keq",
                                k_fwd="p_ADK1_k"))
v_vacpi = jkm.Reaction(
    name="v_VACPI",
    species=['ICPHOS'],
    stoichiometry=[1],
    compartments=['c'],
    mechanism=jcm.Jax_MA_Rev(substrate="ICPHOS",
                             k="p_vacuolePi_k",
                             steady_state_substrate="p_vacuolePi_steadyStatePi")
)

v_amd1 = jkm.Reaction(
    name="v_AMD1",
    species=['ICAMP', 'ICIMP', 'ICATP', 'ICPHOS'],
    stoichiometry=[-1, 1, 0, 0],
    compartments=['c', 'c'],
    mechanism=jcm.Jax_Amd1(substrate="ICAMP",
                           product="ICATP",
                           modifier="ICPHOS",
                           vmax="p_Amd1_Vmax",
                           k50="p_Amd1_K50",
                           ki="p_Amd1_Kpi",
                           k_atp="p_Amd1_Katp", ))
v_ade1312 = jkm.Reaction(
    name="v_ADE1312",
    species=['ICIMP', 'ICAMP'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MA_Irrev(substrate="ICIMP",
                              k_fwd="p_Ade13_Ade12_k"))
v_isn1 = jkm.Reaction(
    name="v_ISN1",
    species=['ICIMP', 'ICPHOS', 'ICINO'],
    stoichiometry=[-1, 1, 1],
    compartments=['c', 'c', 'c'],
    mechanism=jm.Jax_MA_Irrev(substrate="ICIMP",
                              k_fwd="p_Isn1_k"))

v_pnp1 = jkm.Reaction(
    name="v_PNP1",
    species=['ICINO', 'ICHYP'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MA_Irrev(substrate="ICINO",
                              k_fwd="p_Pnp1_k"))

v_hpt1 = jkm.Reaction(
    name="v_HPT1",
    species=['ICIMP', 'ICPHOS', 'ICHYP'],
    stoichiometry=[1, -1, -1],
    compartments=['c', 'c', 'c'],
    mechanism=jm.Jax_MA_Irrev(substrate="ICHYP", k_fwd="p_Hpt1_k"))

#v_pdcs
v_pdc = jkm.Reaction(
    name="v_PDC",
    species=['ICPYR', 'ICACE', 'ICPHOS'],
    stoichiometry=[-1, 1, 0],
    compartments=['c', 'c'],
    mechanism=jcm.Jax_Hill_Irreversible_Inhibition(substrate="ICPYR",
                                                   inhibitor="ICPHOS",
                                                   vmax="p_PDC1_Vmax",
                                                   k_half_substrate="p_PDC1_Kpyr",
                                                   hill="p_PDC1_hill",
                                                   ki="p_PDC1_Kpi", ))
v_mitonadh = jkm.Reaction(
    name="v_MITONADH",
    species=['ICNADH', 'ICNAD'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM(substrate="ICNADH", vmax="p_mitoNADHVmax",
                        km="p_mitoNADHKm"))  # I think this can be replaced by Jax_MM_Irrev_Uni

v_ugp = jkm.Reaction(
    name="v_UGP",
    species=['ICG1P'],
    stoichiometry=[-1],
    compartments=['c'],
    mechanism=jm.Jax_Constant_Flux(v="flux_ugp"))

#modification to mimic the glycolysis model
v_transport_reactions = jkm.Reaction(
    name="v_TRANSPORT",
    species=['ECETOH', 'ECglyc'],
    compartments=['e', 'e'],
    stoichiometry=[-1, -1],
    mechanism=jcm.Jax_Transport_Flux_Correction(substrate="ECETOH", dilution_rate='D')
)


## we can add modifications to mechanism through the JaxKineticModelBuild obejct
compartments = {'c': 1, 'e': 1}
reactions = [v_hxk, v_glt, v_nth1, v_pgi,
             v_sinkg6p, v_sinkf6p, v_pgm1,
             v_tps1, v_tps2, v_pfk, v_ald,
             v_tpi1, v_sinkgap, v_g3pdh,
             v_gapdh, v_pgk, v_sink3pga,
             v_hor2, v_glyct_ic, v_glyct_ec,
             v_pgm, v_eno2, v_sinkpep,
             v_pyk1, v_sinkpyr, v_adh,
             v_sinkace, v_etoht_ic, v_etoht_ec,
             v_atpmito, v_atpase, v_adk,
             v_vacpi, v_amd1, v_ade1312,
             v_isn1, v_pnp1, v_hpt1,
             v_pdc, v_mitonadh, v_ugp,
             v_transport_reactions]

kmodel = jkm.JaxKineticModelBuild(reactions, compartments)


# do an interpolation using sympy for EC glucose and add as a boundary condition
data = pd.read_csv('datasets/VanHeerden_Glucose_Pulse/FF1_timeseries_format.csv', index_col=0)
domain = [float(i) for i in data.loc['ECglucose'].dropna().index]
d_range = data.loc['ECglucose'].dropna().values

t = sympy.Symbol('t')
spline = sympy.interpolating_spline(3, t, domain, d_range)
print('true spline', spline)
spline = str(spline)
kmodel.add_boundary('ECglucose', jkm.BoundaryCondition(spline))





kmodel_sim = jkm.NeuralODEBuild(kmodel)

S = kmodel.stoichiometric_matrix


lit_params = pd.read_csv(('parameter_initializations/'
                          'Glycolysis_model/'
                          'parameter_initialization_glycolysis'
                          '_literature_values.csv'), index_col=0).to_dict()['0']

D = 0.1
parameters = {}
for name in kmodel.parameter_names:
    #need to deal with the poly_sinks dependent on dilution rate
    if name in lit_params.keys():
        parameters[name] = lit_params[name]
    else:
        if name == 'poly_sinkG6P':
            parameters[name] = float(jnp.abs(3.6854 * D ** 3 - 1.4119 * D ** 2 - 0.6312 * D - 0.0043))
        elif name == 'poly_sinkF6P':
            parameters[name] = float(jnp.abs(
                519.3740 * D ** 6 - 447.7990 * D ** 5 + 97.2843 * D ** 4 + 8.0698 * D ** 3
                - 4.4005 * D ** 2 + 0.6254 * D - 0.0078))
        elif name == 'poly_sinkP3G':
            parameters[name] = float(jnp.abs(-0.2381 * D ** 2 - 0.0210 * D - 0.0034))
        elif name == 'poly_sinkPEP':
            parameters[name] = float(jnp.abs(-0.0637 * D ** 2 - 0.0617 * D - 0.0008))
        elif name == 'poly_sinkPYR':
            parameters[name] = float(jnp.abs(
                -8.4853e03 * D ** 6 + 9.4027e03 * D ** 5 - 3.8027e03
                * D ** 4 + 700.5 * D ** 3 - 60.26 * D ** 2 + 0.711 * D - 0.0356))
        elif name == 'poly_sinkACE':
            parameters[name] = float(jnp.abs(
                118.8562 * D ** 6 - 352.3943 * D ** 5 + 245.6092 *
                D ** 4 - 75.2550 * D ** 3 + 11.1153 * D ** 2 - 1.0379 * D + 0.0119))
        elif name == 'ECbiomass':
            parameters[name] = float(3.7683659)
        elif name == "D":
            parameters[name] = float(D)

y0_dict = {
    "ICG1P": 0.064568, "ICT6P": 0.093705, "ICtreh": 63.312040,
    "ICglucose": 0.196003, "ICG6P": 0.716385, "ICF6P": 0.202293,
    "ICFBP": 0.057001, "ICDHAP": 0.048571, "ICG3P": 0.020586,
    "ICglyc": 0.1, "ICGAP": 0.006213, "ICBPG": 0.0001,
    "IC3PG": 2.311074, "IC2PG": 0.297534, "ICPEP": 1.171415,
    "ICPYR": 0.152195, "ICACE": 0.04, "ICETOH": 10.0,
    "ECETOH": 0, "ECglyc": 0.0, "ICNADH": 0.0106,
    "ICNAD": 1.5794, "ICATP": 3.730584, "ICADP": 1.376832,
    "ICAMP": 0.431427, "ICPHOS": 10, "ICIMP": 0.100, "ICINO": 0.100,
    "ICHYP": 1.5, }

y0 = []
for meta in kmodel.species_names:
    if meta in y0_dict.keys():
        y0.append(y0_dict[meta])
    else:
        print(meta)
        y0.append(1)
y0 = jnp.array(y0)



output_dir = "models/manual_implementations/sbml_export/"
model_name = "glycolysis_feastfamine_pulse1"

sbml = SBMLExporter(model=kmodel_sim)

sbml.export(initial_conditions=y0,
            parameters=parameters,
            output_file=f"{output_dir}/{model_name}.xml")

# #
ts = jnp.linspace(0, 300, 1000)
kmodel_sim = jax.jit(kmodel_sim)
ys = kmodel_sim(ts, y0, parameters)
ys = pd.DataFrame(ys, columns=kmodel.species_names)

names = ['ICATP', 'ICADP', 'ICAMP']
for name in names:
    plt.plot(ts, ys[name], label=name)
plt.legend()
plt.show()

names = ['ICT6P', 'ICglucose', 'ICG6P', 'ICF6P']
for name in names:
    plt.plot(ts, ys[name], label=name)
plt.legend()
plt.show()

names = ['ICNADH', 'ICNAD', "ICPHOS"]
for name in names:
    plt.plot(ts, ys[name], label=name)
plt.legend()
plt.show()

names = ['ICT6P', 'ICtreh']
for name in names:
    plt.plot(ts, ys[name], label=name)
plt.legend()
plt.show()

names = ['ICETOH', 'ICPYR', "IC3PG", 'ECETOH']
for name in names:
    plt.plot(ts, ys[name], label=name)
plt.legend()
plt.show()

names = ["IC3PG", "IC2PG", "ICPYR"]
for name in names:
    plt.plot(ts, ys[name], label=name)
plt.legend()
plt.show()

names = ["ICglyc", 'ECglyc']
for name in names:
    plt.plot(ts, ys[name], label=name)
plt.legend()
plt.show()
