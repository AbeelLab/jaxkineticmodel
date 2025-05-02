"""" A documentation script that shows some aspects of the reactions that can be customized
"""
from jaxkineticmodel.kinetic_mechanisms import JaxKineticMechanisms as jm
from jaxkineticmodel.kinetic_mechanisms import JaxKineticModifiers as modifier
from jaxkineticmodel.kinetic_mechanisms.JaxKineticMechanisms import Mechanism


# look into how to deal with arbitrary arguments in compute()
v1 = jm.Jax_MM_Irrev_Uni(
    substrate="A",
    vmax="v_max",
    km_substrate="km_A",
)
print(v1.symbolic())
# A*v_max/(km_A*(A/km_A + 1))
v1.add_modifier(modifier.SimpleActivator(activator="C", k_A="k"))
print(v1.symbolic())
# A*v_max*(C/k + 1)/(km_A*(A/km_A + 1))


class JaxNewMechanism(Mechanism):
    """Example of a custom reaction implementation"""

    @staticmethod
    def compute(substrate, parameter1, parameter2):
        """Here you can add whatever
        mechanistic description you like.
        This is of course a non-sense reaction"""

        return (substrate * (10 / 2) + 5 ** parameter1) / parameter2

v_new = JaxNewMechanism(substrate="A",
                         parameter1="vmax",
                         parameter2="abc_parameter")
print(v_new.symbolic())
# (5**vmax + 5.0*A)/abc_parameter
