#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
"""
Method to set constant pure component properties:

"""
from pyomo.environ import log, Var, units as pyunits
import pyomo.environ as pyo

from idaes.core.util.misc import set_param_from_config
from idaes.core.util.constants import Constants




# -----------------------------------------------------------------------------
# Heat capacities, enthalpies and entropies
class ChapmanEnskogLennardJones(object):
    @staticmethod
    def build_lennard_jones_parameters(cobj):
        units = cobj.parent_block().get_metadata().derived_units
        if not hasattr(cobj, "lennard_jones_sigma"):
            cobj.lennard_jones_sigma = Var(
                doc="Parameter sigma from Lennard Jones potential",
                units=units["length"]
            )
            set_param_from_config(cobj, param="lennard_jones_sigma")

        if not hasattr(cobj, "lennard_jones_epsilon_reduced"):
            cobj.lennard_jones_epsilon_reduced = Var(
                doc="Parameter epsilon from Lennard Jones potential reduced by Boltzmann constant",
                units=units["temperature"]
            )
            set_param_from_config(cobj, param="lennard_jones_epsilon_reduced")

    # Ideal liquid properties methods
    class viscosity_dynamic_vap_comp(object):
        @staticmethod
        def build_parameters(cobj):
            ChapmanEnskogLennardJones.build_lennard_jones_parameters(cobj)
            if not hasattr(cobj, "viscosity_collision_integral_callback"):
                cobj.viscosity_collision_integral_callback = collision_integral_kim_ross_callback

        @staticmethod
        def return_expression(b, cobj, T):
            # Properties of Gases and Liquids, Eq. 9.3.9
            units = b.get_metadata().derived_units

            T = pyunits.convert(T, to_units=pyunits.K)
            sigma = pyunits.convert(cobj.lennard_jones_sigma, pyunits.angstrom)
            M = pyunits.convert(cobj.mw, pyunits.g/pyunits.mol)
            T_dim = T / pyunits.convert(cobj.lennard_jones_epsilon)
            Omega = cobj.viscosity_collision_integral_callback(T_dim)

            C = 26.69 * pyunits.micropoise * pyunits.angstrom ** 2 / pyo.sqrt(pyunits.g / pyunits.mol * pyunits.K)
            visc = C * pyo.sqrt(M * T) / (sigma ** 2 * Omega)
            return pyunits.convert(visc, units["dynamic_viscosity"])

def collision_integral_kim_ross_callback(T_dim):
    # Properties of Gases and Liquids, Eq. 9.4.5
    # T_dim = T^* = T / lennard_jones_epsilon
    return 1.604 / pyo.sqrt(T_dim)

def collision_integral_neufeld_callback(T_dim):
    # Properties of Gases and Liquids, Eq. 9.4.3
    # T_dim = T^* = T / lennard_jones_epsilon
    A = 1.16145
    B = 0.14874
    C = 0.52487
    D = 0.77320
    E = 2.16178
    F = 2.43787
    return A * T_dim ** -B + C * pyo.exp(-D * T_dim) + E * pyo.exp(-F * T_dim)
