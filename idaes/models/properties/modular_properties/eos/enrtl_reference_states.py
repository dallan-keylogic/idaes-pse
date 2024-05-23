#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2023 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""
Reference state sub-methods for eNRTL activity coefficient method.

Only applicable to liquid/electrolyte phases

Reference state classes should implement the following methods:
    - ref_state : expressions for x and X at the reference state
    - ndIdn : derivative term for long-range contribution

References:

Song, Y. and Chen, C.-C., Symmetric Electrolyte Nonrandom Two-Liquid Activity
Coefficient Model, Ind. Eng. Chem. Res., 2009, Vol. 48, pgs. 7788–7797

Rashin, A. A., &#38; Honig, B. (1985). Reevaluation of the Born model of ion hydration.
The Journal of Physical Chemistry, 89(26), 5588–5593.
"""
# TODO: Missing docstrings
# pylint: disable=missing-function-docstring

from pyomo.environ import Expression, Param, units as pyunits, Var

from idaes.core.util.exceptions import BurntToast, ConfigurationError
from idaes.core.util.constants import Constants
from idaes.core.util.misc import set_param_from_config
# from idaes.models.properties.modular_properties.base.generic_property import set_param_value
from idaes.models.properties.modular_properties.base.utility import (
    get_method,
    get_component_object as cobj,
)
import idaes.logger as idaeslog

# Set up logger
_log = idaeslog.getLogger(__name__)


# Near-zero value to use for unsymmetric reference state
EPS = 1e-20


class Unsymmetric(object):
    """
    Sub-methods for the infinite dilution mixed-solvent reference state
    """
    @staticmethod
    def build_parameters(b):
        b.eps_eNRTL = Param(initialize=EPS, mutable=True)

    @staticmethod
    def ref_state(b, pname):
        def rule_x_ref(b, i):
            pobj = b.params.get_phase(pname)
            eps = getattr(pobj, "eps_eNRTL")
            if i in b.params.solvent_set or i in b.params.solute_set:
                # Eqn 66
                return b.mole_frac_phase_comp_true[pname, i] / sum(
                    b.mole_frac_phase_comp_true[pname, j]
                    for j in b.params.solvent_set | b.params.solute_set
                )
            else:
                return eps

        b.add_component(
            pname + "_x_ref", Expression(b.params.true_species_set, rule=rule_x_ref)
        )
        _ref_shebang(b, pname)

    @staticmethod
    def ndIdn(b, pname, i):
        dimensionless_zero = getattr(b, pname + "_dimensionless_zero")
        # Eqn 71
        return dimensionless_zero


class Symmetric(object):
    """
    Sub-methods for the symmetric (fused-salt) reference state
    """
    @staticmethod
    def build_parameters(b):
        pass

    @staticmethod
    def ref_state(b, pname):
        def rule_x_ref(b, i):
            dimensionless_zero = getattr(b, pname + "_dimensionless_zero")
            if i in b.params.ion_set:
                # Eqn 66
                return b.mole_frac_phase_comp_true[pname, i] / sum(
                    b.mole_frac_phase_comp_true[pname, j] for j in b.params.ion_set
                )
            else:
                return dimensionless_zero

        b.add_component(
            pname + "_x_ref", Expression(b.params.true_species_set, rule=rule_x_ref)
        )
        _ref_shebang(b, pname)

    @staticmethod
    def ndIdn(b, pname, i):
        # Eqn 75
        return 0.5 * sum(
            cobj(b, j).config.charge ** 2 * ndxdn(b, pname, i, j)
            for j in b.params.ion_set
        )

class InfiniteDilutionSingleSolvent(object):
    """
    Sub-methods for the infinite dilution single solvent reference state
    """
    @staticmethod
    def build_parameters(b):
        params = b.parent_block()
        for j in params.ion_set:
            cobj = params.get_component(j)
            units = cobj.parent_block().get_metadata().derived_units
            # Rashin and Honig (1985) recommend 1.07 times the ion's "covalent radius"
            # in a vacuum if a hydrogen-bonding solvent is used. In principle, this
            # radius is different in different solvents, but in practice we don't
            # data to correlate it
            if not hasattr(cobj, "born_radius"):
                cobj.born_radius = Var(
                    doc="Cavity radius from Born model of solvation",
                    units=units["length"],
                )
                set_param_from_config(cobj, param="born_radius")
        
        ref_comp = b.config["equation_of_state_options"]["reference_component"]
        if ref_comp not in params.component_list:
            return ConfigurationError(
                f"Specified component for infinite dilution reference state {ref_comp} is not a component in the mixture. "
                "Check if it is misspelled."
            )
        if ref_comp not in params.solvent_set:
            return ConfigurationError(f"Specified component for infinite dilution reference state {ref_comp} is not a solvent.")
        b.ref_comp = ref_comp

    @staticmethod
    def ref_state(b, pname):
        # No ions at infinite dilution, ionic strength is zero

        def rule_I_ref(b):
            dimensionless_zero = getattr(b, pname + "_dimensionless_zero")
            return dimensionless_zero
        
        b.add_component(
            pname + "_ionic_strength_ref",
            Expression(rule=rule_I_ref, doc="Ionic strength at reference state"),
        )

        def rule_log_gamma_born(b, s):
            pobj = b.params.get_phase(pname)
            ref_comp = pobj.ref_comp
            dimensionless_zero = getattr(b, pname + "_dimensionless_zero")

            if not s in b.params.ion_set:
                return dimensionless_zero
            if len(b.params.solvent_set) == 1:
                return dimensionless_zero
            q_e = Constants.elemental_charge
            k_b = Constants.boltzmann_constant
            eps_vacuum = Constants.vacuum_electric_permittivity
            pi = Constants.pi
            eps_ref_comp =  get_method(b, "relative_permittivity_liq_comp", ref_comp)(
                b, cobj(b, ref_comp), b.temperature
            )
            eps_solvent = getattr(b, pname + "_relative_permittivity_solvent")
            z = abs(cobj(b, s).config.charge)
            r = cobj(b, s).born_radius

            return (z*q_e)**2/(8 * pi *eps_vacuum * k_b * b.temperature * r) * (1/eps_solvent - 1/eps_ref_comp)
        
        b.add_component(
            pname + "_log_gamma_born",
            Expression(
                b.params.true_species_set,
                rule=rule_log_gamma_born,
                doc="Born term contribution to activity coefficient"
            ),
        )
        def rule_enth_mol_excess_born(b):
            units = b.params.get_metadata().derived_units
            q_e = Constants.elemental_charge
            eps_vacuum = Constants.vacuum_electric_permittivity
            pi = Constants.pi
            N_A = Constants.avogadro_number
            T = pyunits.convert(b.temperature, to_units=pyunits.K)
            eps_solvent = getattr(b, pname + "_relative_permittivity_solvent")
            d_eps_solvent_dT = getattr(b, pname + "_d_relative_permittivity_solvent_dT")

            if len(b.params.ion_set) == 0:
                return 0 * units.ENERGY/units.AMOUNT
            else:
                return pyunits.convert(
                    N_A * (q_e)**2/(8 * pi * eps_vacuum * eps_solvent)
                    * (1 + T / eps_solvent * d_eps_solvent_dT)
                    * sum(
                        b.mole_frac_phase_comp_true[pname, s]
                        * abs(cobj(b, s).config.charge) ** 2
                        / pyunits.convert(
                            cobj(b, s).born_radius,
                            to_units=pyunits.m
                        )
                        for s in b.params.ion_set
                    ),
                    to_units=units.ENERGY/units.AMOUNT
                )
        
        b.add_component(
            pname + "_enth_mol_excess_born",
            Expression(
                rule=rule_enth_mol_excess_born,
                doc="Excess enthalpy contribution from Born term"
            ),
        )

        def rule_log_gamma_lc_I0(b, s):
            pobj = b.params.get_phase(pname)
            ref_comp = pobj.ref_comp
            dimensionless_zero = getattr(b, pname + "_dimensionless_zero")

            G = getattr(b, pname + "_G")
            tau = getattr(b, pname + "_tau")
            if s in b.params.ion_set:
                # Eqn. 54/55 
                # The activity of this "reference state" changes with changing ionic composition
                # of the undiluted mixture (through using Y_a/Y_c to calculate G_a/c,H2O). This
                # is probably a bad thing, but everyone else calculates things this way so we might
                # as well calculate things this way too.
                Z = abs(b.params.get_component(s).config.charge)
                return Z * (G[s, ref_comp]*tau[s, ref_comp] + tau[ref_comp, s])
            elif s in b.params.solute_set:
                return G[s, ref_comp]*tau[s, ref_comp] + tau[ref_comp, s]
            elif s in b.params.solvent_set or s in b.params.zwitterion_set:
                return dimensionless_zero
            else:
                raise BurntToast(
                    f"{s} eNRTL model encountered unexpected component."
                )

        b.add_component(
            pname + "_log_gamma_lc_I0",
            Expression(
                b.params.true_species_set,
                rule=rule_log_gamma_lc_I0,
                doc="Local contribution at reference state",
            ),
        )
        def rule_d_log_gamma_lc_I0_dT(b, s):
            pobj = b.params.get_phase(pname)
            ref_comp = pobj.ref_comp
            dimensionless_zero = getattr(b, pname + "_dimensionless_zero")
            units = b.params.get_metadata().derived_units

            G = getattr(b, pname + "_G")
            dG_dT = getattr(b, pname + "_dG_dT")
            tau = getattr(b, pname + "_tau")
            dtau_dT = getattr(b, pname + "_dtau_dT")
            if s in b.params.ion_set:
                # Eqn. 54/55 
                # The activity of this "reference state" changes with changing ionic composition
                # of the undiluted mixture (through using Y_a/Y_c to calculate G_a/c,H2O). This
                # is probably a bad thing, but everyone else calculates things this way so we might
                # as well calculate things this way too.
                Z = abs(b.params.get_component(s).config.charge)
                return Z * (
                    dG_dT[s, ref_comp] * tau[s, ref_comp]
                    + G[s, ref_comp] * dtau_dT[s, ref_comp] 
                    + dtau_dT[ref_comp, s]
                )
            elif s in b.params.solute_set:
                return (
                    dG_dT[s, ref_comp] * tau[s, ref_comp]
                    + G[s, ref_comp] * dtau_dT[s, ref_comp] 
                    + dtau_dT[ref_comp, s]
                )
            elif s in b.params.solvent_set or s in b.params.zwitterion_set:
                return dimensionless_zero / units.TEMPERATURE
            else:
                raise BurntToast(
                    f"{s} eNRTL model encountered unexpected component."
                )

        b.add_component(
            pname + "_d_log_gamma_lc_I0_dT",
            Expression(
                b.params.true_species_set,
                rule=rule_d_log_gamma_lc_I0_dT,
                doc="Local contribution at reference state",
            ),
        )
    @staticmethod
    def ndIdn(b, pname, i):
        dimensionless_zero = getattr(b, pname + "_dimensionless_zero")
        # Eqn 71
        return dimensionless_zero


def ndxdn(b, pname, i, j):
    x0 = getattr(b, pname + "_x_ref")

    # Delta function used in Eqns 73-76 (not defined in paper)
    if i == j:
        delta = 1.0
    else:
        delta = 0.0

    # Eqn 76
    return (delta - x0[j]) / sum(
        b.mole_frac_phase_comp_true[pname, k] for k in b.params.ion_set
    )

def _ref_shebang(b, pname):
    """
    Construct expressions necessary to evaluate activity coefficients at reference state
    """
    # Defer import to avoid circular import
    from idaes.models.properties.modular_properties.eos.enrtl import log_gamma_lc

    dimensionless_zero = getattr(b, pname + "_dimensionless_zero")

    def rule_I_ref(b):  # Eqn 62 evaluated at reference state
        x = getattr(b, pname + "_x_ref")
        return 0.5 * sum(
            x[c] * b.params.get_component(c).config.charge ** 2
            for c in b.params.ion_set
        )

    b.add_component(
        pname + "_ionic_strength_ref",
        Expression(rule=rule_I_ref, doc="Ionic strength at reference state"),
    )

    def rule_X_ref(b, j):  # Eqn 21 evaluated at reference state
        x = getattr(b, pname + "_x_ref")
        if (pname, j) not in b.params.true_phase_component_set:
            return Expression.Skip
        elif j in b.params.ion_set:
            return x[j] * abs(cobj(b, j).config.charge)
        else:
            return x[j]

    b.add_component(
        pname + "_X_ref",
        Expression(
            b.params.true_species_set,
            rule=rule_X_ref,
            doc="Charge x mole fraction term at reference state",
        ),
    )

    def rule_log_gamma_lc_I0(b, s):
        X = getattr(b, pname + "_X_ref")
        G = getattr(b, pname + "_G")
        tau = getattr(b, pname + "_tau")
        if not s in b.params.ion_set:
            return dimensionless_zero
        else:
            return log_gamma_lc(b, pname, s, X, G, tau)

    b.add_component(
        pname + "_log_gamma_lc_I0",
        Expression(
            b.params.true_species_set,
            rule=rule_log_gamma_lc_I0,
            doc="Local contribution at reference state",
        ),
    )

    def rule_log_gamma_born(b, s):
        return dimensionless_zero
    b.add_component(
        pname + "_log_gamma_born",
        Expression(
            b.params.true_species_set,
            rule=rule_log_gamma_born,
            doc="Local contribution at reference state",
        )

    )