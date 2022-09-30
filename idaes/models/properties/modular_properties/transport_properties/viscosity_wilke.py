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
from idaes.models.properties.modular_properties.base.utility import get_component_object




# ------------------------------------------------------------------------------------
# Gas viscosity at low pressures from Lennard Jones paramters. Note that LJ parameters
# are underdetermined when estimated from viscosity data (see The Indeterminacy of
# the Values of Potential Parameters as Derived from Transport and Virial Coefficient
# by Reichenberg D., 1973 for more information) so it's important to use LJ parameters
# from the same source.
class ViscosityWilkePhase(object):
    @staticmethod
    def build_parameters(pobj):
        if not hasattr(pobj, "viscosity_phi_ij_callback"):
            pobj.viscosity_phi_ij_callback = wilke_phi_ij_callback
    @staticmethod
    def build_phi_ij(b, pobj):
        pname = pobj.local_name
        if not hasattr(b, "viscosity_phi_ij"):
            mw_dict = {k: b.params.get_component(k).mw for k in b.components_in_phase(pname)}

            def phi_rule(blk, i, j):
                return pobj.viscosity_phi_ij_callback(blk, i, j, pname, mw_dict)

            b.visc_phi_ij = pyo.Expression(
                b.components_in_phase(pname),
                b.components_in_phase(pname),
                rule=phi_rule,
                doc="Intermediate quantity for calculating gas mixture viscosity and thermal conductivity"
            )

    class visc_d(object):
        @staticmethod
        def build_parameters(pobj):
            ViscosityWilkePhase.build_parameters(pobj)

        @staticmethod
        def return_expression(b, pobj):
            # Properties of Gases and Liquids, Eq. 9-5.14
            ViscosityWilkePhase.build_phi_ij(b, pobj)

            pname = pobj.local_name

            return sum([b.mole_frac_phase_comp[pname, i] * b.visc_d_phase_comp[pname, i]
                        / sum([b.mole_frac_phase_comp[pname, j] * b.visc_phi_ij[i, j] for j in b.components_in_phase(pname)])
                        for i in b.components_in_phase(pname)])


def wilke_phi_ij_callback(b, i, j, pname, mw_dict):
    # Equation 9-5.14 in Properties of Gases and Liquids 5th ed.
    visc_i = b.visc_d_phase_comp[pname, i]
    visc_j = b.visc_d_phase_comp[pname, j]
    return (
        (1 + pyo.sqrt(visc_i/visc_j)
            * (mw_dict[j]/mw_dict[i]) ** 0.25) ** 2
        / pyo.sqrt(8 * (1 + mw_dict[i]/mw_dict[j]))
    )

def herring_zimmer_phi_ij_callback(b, i, j, pname, mw_dict):
    # Equation 9-5.17 in Properties of Gases and Liquids 5th ed.
    return pyo.sqrt(mw_dict[j] / mw_dict[i])