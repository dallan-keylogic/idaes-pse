#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""
Methods for eNRTL activity coefficient method.

Only applicable to liquid/electrolyte phases

Many thanks to C.-C. Chen for his assistance and suggestions on testing and
verifying the model.

Reference:

Song, Y. and Chen, C.-C., Symmetric Electrolyte Nonrandom Two-Liquid Activity
Coefficient Model, Ind. Eng. Chem. Res., 2009, Vol. 48, pgs. 7788–7797

Note that "charge number" in the paper refers to the absolute value of the
ionic charge.
"""
# TODO: Missing docstrings
# pylint: disable=missing-function-docstring

# TODO: Look into protected access issues
# pylint: disable=protected-access

from enum import Enum
from functools import partial

from pyomo.environ import Expression, exp, log, Param, Set, units as pyunits
from pyomo.core.expr.calculus.derivatives import Modes, differentiate

from idaes.core.util.exceptions import PropertyNotSupportedError
from idaes.models.properties.modular_properties.base.utility import (
    get_method,
    get_component_object as cobj,
)
from idaes.models.properties.modular_properties.base.generic_property import StateIndex
from idaes.models.properties.modular_properties.phase_equil.henry import HenryType
from idaes.core.util.constants import Constants
from idaes.core.util.exceptions import ConfigurationError, BurntToast
import idaes.logger as idaeslog

from idaes.models.properties.modular_properties.phase_equil.henry import (
    henry_pressure,
    log_henry_pressure,
)

from .ideal import Ideal
from .enrtl_reference_states import Symmetric
from .enrtl_parameters import ConstantAlpha, ConstantTau


# Set up logger
_log = idaeslog.getLogger(__name__)


DefaultAlphaRule = ConstantAlpha
DefaultTauRule = ConstantTau
DefaultRefState = Symmetric

class EnthMolPhaseBasis(Enum):
    """Enum for method of calculating enth_mol_phase"""
    true = 0
    apparent = 1


# Closest approach parameter - implemented as a global constant for now
# This is not something the user should be changing in most cases
ClosestApproach = 14.9


class ENRTL(Ideal):
    """EoS class for eNRTL based property packages."""

    # Add attribute indicating support for electrolyte systems
    electrolyte_support = True

    @staticmethod
    def build_parameters(b):
        # Build additional indexing sets
        pblock = b.parent_block()
        ion_pair = []
        for i in pblock.cation_set:
            for j in pblock.anion_set:
                ion_pair.append(i + ", " + j)
        b.ion_pair_set = Set(initialize=ion_pair)

        comps = pblock.solvent_set | pblock.solute_set | b.ion_pair_set | pblock.zwitterion_set
        comp_pairs = []
        comp_pairs_sym = []
        for i in comps:
            for j in comps:
                if i in pblock.solvent_set | pblock.solute_set or i != j:
                    comp_pairs.append((i, j))
                    if (j, i) not in comp_pairs_sym:
                        comp_pairs_sym.append((i, j))
        b.component_pair_set = Set(initialize=comp_pairs)
        b.component_pair_set_symmetric = Set(initialize=comp_pairs_sym)

        # Check options for alpha rule
        if (
            b.config.equation_of_state_options is not None
            and "alpha_rule" in b.config.equation_of_state_options
        ):
            b.config.equation_of_state_options["alpha_rule"].build_parameters(b)
        else:
            DefaultAlphaRule.build_parameters(b)

        # Check options for tau rule
        if (
            b.config.equation_of_state_options is not None
            and "tau_rule" in b.config.equation_of_state_options
        ):
            b.config.equation_of_state_options["tau_rule"].build_parameters(b)
        else:
            DefaultTauRule.build_parameters(b)

        if (
            b.config.equation_of_state_options is not None
            and "reference_state" in b.config.equation_of_state_options
        ):
            b._reference_state_enrtl = b.config.equation_of_state_options["reference_state"]
        else:
            b._reference_state_enrtl = DefaultRefState
        
        b._reference_state_enrtl.build_parameters(b)


    @staticmethod
    def common(b, pobj):
        # For nonaqueous phases, we don't want to build all the eNRTL infrastructure
        # Theoretically in the future we could use NRTL for organic phases
        if not pobj.is_aqueous_phase():
            Ideal.common(b, pobj)
            return
        pname = pobj.local_name
        units = b.params.get_metadata().derived_units

        # For the purposes of eNRTL, zwitterions are "molecules" not "ions"
        molecular_set = b.params.solvent_set | b.params.solute_set | b.params.zwitterion_set

        # Check options for alpha rule
        if (
            pobj.config.equation_of_state_options is not None
            and "alpha_rule" in pobj.config.equation_of_state_options
        ):
            alpha_rule = pobj.config.equation_of_state_options[
                "alpha_rule"
            ].return_alpha_expression
            dalpha_dT_rule = pobj.config.equation_of_state_options[
                "alpha_rule"
            ].return_dalpha_dT_expression
        else:
            alpha_rule = DefaultAlphaRule.return_alpha_expression
            dalpha_dT_rule = DefaultAlphaRule.return_dalpha_dT_expression

        # Check options for tau rule
        if (
            pobj.config.equation_of_state_options is not None
            and "tau_rule" in pobj.config.equation_of_state_options
        ):
            tau_rule = pobj.config.equation_of_state_options[
                "tau_rule"
            ].return_tau_expression
            dtau_dT_rule = pobj.config.equation_of_state_options[
                "tau_rule"
            ].return_dtau_dT_expression
        else:
            tau_rule = DefaultTauRule.return_tau_expression
            dtau_dT_rule = DefaultTauRule.return_dtau_dT_expression

        # Check options for reference state
        if (
            pobj.config.equation_of_state_options is not None
            and "reference_state" in pobj.config.equation_of_state_options
        ):
            ref_state = pobj.config.equation_of_state_options["reference_state"]
        else:
            ref_state = DefaultRefState

        # ---------------------------------------------------------------------
        # Calculate composition terms
        dimensionless_zero = Param(
            initialize=0.0,
            mutable=False,
            doc="Dimensionless zero to avoid issue with AD (Pyomo 6.6.2)"
        )
        b.add_component(
            pname + "_dimensionless_zero",
            dimensionless_zero,
        )
        # Ionic Strength
        def rule_I(b):  # Eqn 62
            dimensionless_zero = getattr(b, pname + "_dimensionless_zero")
            if len(b.params.ion_set) > 0:
                return 0.5 * sum(
                    b.mole_frac_phase_comp_true[pname, c]
                    * b.params.get_component(c).config.charge ** 2
                    for c in b.params.ion_set
                ) 
            else:
                return dimensionless_zero # TODO get rid of this once Pyomo fixes their AD

        b.add_component(
            pname + "_ionic_strength", Expression(rule=rule_I, doc="Ionic strength")
        )

        # Calculate mixing factors
        def rule_X(b, j):  # Eqn 21
            if (pname, j) not in b.params.true_phase_component_set:
                return Expression.Skip
            elif j in b.params.cation_set or j in b.params.anion_set:
                return b.mole_frac_phase_comp_true[pname, j] * abs(
                    cobj(b, j).config.charge
                )
            else:
                return b.mole_frac_phase_comp_true[pname, j]

        b.add_component(
            pname + "_X",
            Expression(
                b.params.true_species_set,
                rule=rule_X,
                doc="Charge x mole fraction term",
            ),
        )

        def rule_Y(b, j):
            if cobj(b, j).config.charge < 0:
                # Anion
                dom = b.params.anion_set
            else:
                dom = b.params.cation_set

            X = getattr(b, pname + "_X")
            return X[j] / sum(X[i] for i in dom)  # Eqns 36 and 37

        # Y is a charge ratio, and thus independent of x for symmetric state
        # TODO: This may need to change for the unsymmetric state
        b.add_component(
            pname + "_Y",
            Expression(b.params.ion_set, rule=rule_Y, doc="Charge composition"),
        )

        # ---------------------------------------------------------------------
        # Long-range terms
        # Average molar volume of solvent
        def rule_vol_mol_solvent(b):  # Eqn 77
            if len(b.params.solvent_set) == 1:
                s = b.params.solvent_set.first()
                return ENRTL.get_vol_mol_pure(b, "liq", s, b.temperature)
            else:
                return sum(
                    b.mole_frac_phase_comp_true[pname, s]
                    * ENRTL.get_vol_mol_pure(b, "liq", s, b.temperature)
                    for s in b.params.solvent_set
                ) / sum(
                    b.mole_frac_phase_comp_true[pname, s] for s in b.params.solvent_set
                )

        b.add_component(
            pname + "_vol_mol_solvent",
            Expression(rule=rule_vol_mol_solvent, doc="Mean molar volume of solvent"),
        )

        # Mean relative permitivity of solvent
        def rule_eps_solvent(b):  # Eqn 78
            if len(b.params.solvent_set) == 1:
                s = b.params.solvent_set.first()
                return get_method(b, "relative_permittivity_liq_comp", s)(
                    b, cobj(b, s), b.temperature
                )
            else:
                return sum(
                    b.mole_frac_phase_comp_true[pname, s]
                    * get_method(b, "relative_permittivity_liq_comp", s)(
                        b, cobj(b, s), b.temperature
                    )
                    * b.params.get_component(s).mw
                    for s in b.params.solvent_set
                ) / sum(
                    b.mole_frac_phase_comp_true[pname, s] * b.params.get_component(s).mw
                    for s in b.params.solvent_set
                )

        b.add_component(
            pname + "_relative_permittivity_solvent",
            Expression(
                rule=rule_eps_solvent, doc="Mean relative permittivity  of solvent"
            ),
        )

        def rule_d_eps_solvent_dT(b):
            eps_solvent = getattr(b, pname + "_relative_permittivity_solvent")
            return differentiate(eps_solvent, b.temperature, mode=Modes.reverse_symbolic)
        
        b.add_component(
            pname + "_d_relative_permittivity_solvent_dT",
            Expression(
                rule=rule_d_eps_solvent_dT, doc="Temperature derivative of solvent relative permittivity"
            ),
        )

        # Debye-Huckel parameter
        def rule_A_DH(b):  # Eqn 61
            # Note: Where the paper refers to the dielectric constant, it
            # actually means the electric permittivity of the solvent
            # eps = eps_r*eps_0 (units F/m)
            # Note that paper is missing a required 4*pi term
            v = pyunits.convert(
                getattr(b, pname + "_vol_mol_solvent"), pyunits.m**3 / pyunits.mol
            )
            eps = getattr(b, pname + "_relative_permittivity_solvent")
            eps0 = Constants.vacuum_electric_permittivity
            return (
                (1 / 3)
                * (2 * Constants.pi * Constants.avogadro_number / v) ** 0.5
                * (
                    Constants.elemental_charge**2
                    / (
                        4
                        * Constants.pi
                        * eps
                        * eps0
                        * Constants.boltzmann_constant
                        * b.temperature
                    )
                )
                ** (3 / 2)
            )

        b.add_component(
            pname + "_A_DH", Expression(rule=rule_A_DH, doc="Debye-Huckel parameter")
        )

        # ---------------------------------------------------------------------
        # Local Contribution Terms
        # For the symmetric state, all of these are independent of composition
        # TODO: For the unsymmetric state, it may be necessary to recalculate
        # Calculate alphas for all true species pairings
        b.add_component(
            pname + "_alpha_default",
            Param(initialize=0.2, mutable=False, doc="Constant value of Alpha used in eNRTL"),
        )

        def rule_alpha_expr(b, i, j):
            Y = getattr(b, pname + "_Y")
            alpha_default = getattr(b, pname + "_alpha_default")

            if (pname, i) not in b.params.true_phase_component_set or (
                pname,
                j,
            ) not in b.params.true_phase_component_set:
                return Expression.Skip
            elif (i in molecular_set) and (j in molecular_set):
                # alpha equal user provided parameters
                return alpha_rule(b, pobj, i, j, b.temperature)
            elif i in b.params.cation_set and j in molecular_set:
                # Eqn 32
                return sum(
                    Y[k] * alpha_rule(b, pobj, (i + ", " + k), j, b.temperature)
                    for k in b.params.anion_set
                )
            elif j in b.params.cation_set and i in molecular_set:
                # Eqn 32
                return sum(
                    Y[k] * alpha_rule(b, pobj, (j + ", " + k), i, b.temperature)
                    for k in b.params.anion_set
                )
            elif i in b.params.anion_set and j in molecular_set:
                # Eqn 33
                return sum(
                    Y[k] * alpha_rule(b, pobj, (k + ", " + i), j, b.temperature)
                    for k in b.params.cation_set
                )
            elif j in b.params.anion_set and i in molecular_set:
                # Eqn 33
                return sum(
                    Y[k] * alpha_rule(b, pobj, (k + ", " + j), i, b.temperature)
                    for k in b.params.cation_set
                )
            elif i in b.params.cation_set and j in b.params.anion_set:
                # Eqn 34
                if len(b.params.cation_set) > 1:
                    return sum(
                        Y[k]
                        * alpha_rule(
                            b, pobj, (i + ", " + j), (k + ", " + j), b.temperature
                        )
                        for k in b.params.cation_set
                    )
                else:
                    return alpha_default
            elif i in b.params.anion_set and j in b.params.cation_set:
                # Eqn 35
                if len(b.params.anion_set) > 1:
                    return sum(
                        Y[k]
                        * alpha_rule(
                            b, pobj, (j + ", " + i), (j + ", " + k), b.temperature
                        )
                        for k in b.params.anion_set
                    )
                else:
                    return alpha_default
            elif (i in b.params.cation_set and j in b.params.cation_set) or (
                i in b.params.anion_set and j in b.params.anion_set
            ):
                # No like-ion interactions
                return Expression.Skip
            else:
                raise BurntToast(
                    "{} eNRTL model encountered unexpected component pair {}.".format(
                        b.name, (i, j)
                    )
                )

        b.add_component(
            pname + "_alpha",
            Expression(
                b.params.true_species_set,
                b.params.true_species_set,
                rule=rule_alpha_expr,
                doc="Non-randomness parameters",
            ),
        )

        def rule_dalpha_dT_expr(b, i, j):
            Y = getattr(b, pname + "_Y")

            if (pname, i) not in b.params.true_phase_component_set or (
                pname,
                j,
            ) not in b.params.true_phase_component_set:
                return Expression.Skip
            elif (i in molecular_set) and (j in molecular_set):
                # alpha equal user provided parameters
                return dalpha_dT_rule(b, pobj, i, j, b.temperature)
            elif i in b.params.cation_set and j in molecular_set:
                # Eqn 32
                return sum(
                    Y[k] * dalpha_dT_rule(b, pobj, (i + ", " + k), j, b.temperature)
                    for k in b.params.anion_set
                )
            elif j in b.params.cation_set and i in molecular_set:
                # Eqn 32
                return sum(
                    Y[k] * dalpha_dT_rule(b, pobj, (j + ", " + k), i, b.temperature)
                    for k in b.params.anion_set
                )
            elif i in b.params.anion_set and j in molecular_set:
                # Eqn 33
                return sum(
                    Y[k] * dalpha_dT_rule(b, pobj, (k + ", " + i), j, b.temperature)
                    for k in b.params.cation_set
                )
            elif j in b.params.anion_set and i in molecular_set:
                # Eqn 33
                return sum(
                    Y[k] * dalpha_dT_rule(b, pobj, (k + ", " + j), i, b.temperature)
                    for k in b.params.cation_set
                )
            elif i in b.params.cation_set and j in b.params.anion_set:
                # Eqn 34
                if len(b.params.cation_set) > 1:
                    return sum(
                        Y[k]
                        * dalpha_dT_rule(
                            b, pobj, (i + ", " + j), (k + ", " + j), b.temperature
                        )
                        for k in b.params.cation_set
                    )
                else:
                    return 0 / units.TEMPERATURE
            elif i in b.params.anion_set and j in b.params.cation_set:
                # Eqn 35
                if len(b.params.anion_set) > 1:
                    return sum(
                        Y[k]
                        * dalpha_dT_rule(
                            b, pobj, (j + ", " + i), (j + ", " + k), b.temperature
                        )
                        for k in b.params.anion_set
                    )
                else:
                    return 0 / units.TEMPERATURE
            elif (i in b.params.cation_set and j in b.params.cation_set) or (
                i in b.params.anion_set and j in b.params.anion_set
            ):
                # No like-ion interactions
                return Expression.Skip
            else:
                raise BurntToast(
                    "{} eNRTL model encountered unexpected component pair {}.".format(
                        b.name, (i, j)
                    )
                )
        b.add_component(
            pname + "_dalpha_dT",
            Expression(
                b.params.true_species_set,
                b.params.true_species_set,
                rule=rule_dalpha_dT_expr,
                doc="Non-randomness parameters",
            ),
        )
        
        # Calculate G terms
        b.add_component(
            pname + "_G_default",
            Param(initialize=1.0, doc="Constant value of G used in eNRTL"),
        )

        def _G_appr(b, pobj, i, j, T):  # Eqn 23
            pname = pobj.local_name
            G_default = getattr(b, pname + "_G_default")
            if i != j:
                return exp(
                    -alpha_rule(b, pobj, i, j, T) * tau_rule(b, pobj, i, j, T)
                )
            else:
                return G_default

        def rule_G_expr(b, i, j):
            Y = getattr(b, pname + "_Y")
            G_default = getattr(b, pname + "_G_default")

            if (pname, i) not in b.params.true_phase_component_set or (
                pname,
                j,
            ) not in b.params.true_phase_component_set:
                return Expression.Skip
            elif (i in molecular_set) and (j in molecular_set):
                # G comes directly from parameters
                return _G_appr(b, pobj, i, j, b.temperature)
            elif i in b.params.cation_set and j in molecular_set:
                # Eqn 38
                return sum(
                    Y[k] * _G_appr(b, pobj, (i + ", " + k), j, b.temperature)
                    for k in b.params.anion_set
                )
            elif i in molecular_set and j in b.params.cation_set:
                # Eqn 40
                return sum(
                    Y[k] * _G_appr(b, pobj, i, (j + ", " + k), b.temperature)
                    for k in b.params.anion_set
                )
            elif i in b.params.anion_set and j in molecular_set:
                # Eqn 39
                return sum(
                    Y[k] * _G_appr(b, pobj, (k + ", " + i), j, b.temperature)
                    for k in b.params.cation_set
                )
            elif i in molecular_set and j in b.params.anion_set:
                # Eqn 41
                return sum(
                    Y[k] * _G_appr(b, pobj, i, (k + ", " + j), b.temperature)
                    for k in b.params.cation_set
                )
            elif i in b.params.cation_set and j in b.params.anion_set:
                # Eqn 42
                if len(b.params.cation_set) > 1:
                    return sum(
                        Y[k]
                        * _G_appr(
                            b, pobj, (i + ", " + j), (k + ", " + j), b.temperature
                        )
                        for k in b.params.cation_set
                    )
                else:
                    # This term does not exist for single cation systems
                    # However, need a valid result to calculate tau
                    return G_default
            elif i in b.params.anion_set and j in b.params.cation_set:
                # Eqn 43
                if len(b.params.anion_set) > 1:
                    return sum(
                        Y[k]
                        * _G_appr(
                            b, pobj, (j + ", " + i), (j + ", " + k), b.temperature
                        )
                        for k in b.params.anion_set
                    )
                else:
                    # This term does not exist for single anion systems
                    # However, need a valid result to calculate tau
                    return G_default
            elif (i in b.params.cation_set and j in b.params.cation_set) or (
                i in b.params.anion_set and j in b.params.anion_set
            ):
                # No like-ion interactions
                return Expression.Skip
            else:
                raise BurntToast(
                    "{} eNRTL model encountered unexpected component pair {}.".format(
                        b.name, (i, j)
                    )
                )

        b.add_component(
            pname + "_G",
            Expression(
                b.params.true_species_set,
                b.params.true_species_set,
                rule=rule_G_expr,
                doc="Local interaction G term",
            ),
        )

        b.add_component(
            pname + "_dG_dT_default",
            Param(
                initialize=0.0,
                mutable=True, # Mutable has to be true for Params with units
                units=1/units.TEMPERATURE,
                doc="Derivative of constant value of G used in eNRTL"
            ),
        )

        def rule_dG_dT_expr(b, i, j):
            Y = getattr(b, pname + "_Y")
            G = getattr(b, pname+"_G")
            dG_dT_default = getattr(b, pname + "_dG_dT_default")
                
            def _dG_dT_appr(b, pobj, i, j, T):
                if i != j:
                    return -_G_appr(
                        b, pobj, i, j, b.temperature
                    ) * (
                        dalpha_dT_rule(b, pobj, i, j, T) * tau_rule(b, pobj, i, j, T)
                        + alpha_rule(b, pobj, i, j, T) * dtau_dT_rule(b, pobj, i, j, T)
                    )
                else:
                    return dG_dT_default

            if (pname, i) not in b.params.true_phase_component_set or (
                pname,
                j,
            ) not in b.params.true_phase_component_set:
                return Expression.Skip
            elif (i in molecular_set) and (j in molecular_set):
                # G comes directly from parameters
                return _dG_dT_appr(b, pobj, i, j, b.temperature)
            elif i in b.params.cation_set and j in molecular_set:
                # Eqn 38
                return sum(
                    Y[k] * _dG_dT_appr(b, pobj, (i + ", " + k), j, b.temperature)
                    for k in b.params.anion_set
                )
            elif i in molecular_set and j in b.params.cation_set:
                # Eqn 40
                return sum(
                    Y[k] * _dG_dT_appr(b, pobj, i, (j + ", " + k), b.temperature)
                    for k in b.params.anion_set
                )
            elif i in b.params.anion_set and j in molecular_set:
                # Eqn 39
                return sum(
                    Y[k] * _dG_dT_appr(b, pobj, (k + ", " + i), j, b.temperature)
                    for k in b.params.cation_set
                )
            elif i in molecular_set and j in b.params.anion_set:
                # Eqn 41
                return sum(
                    Y[k] * _dG_dT_appr(b, pobj, i, (k + ", " + j), b.temperature)
                    for k in b.params.cation_set
                )
            elif i in b.params.cation_set and j in b.params.anion_set:
                # Eqn 42
                if len(b.params.cation_set) > 1:
                    return sum(
                        Y[k]
                        * _dG_dT_appr(
                            b, pobj, (i + ", " + j), (k + ", " + j), b.temperature
                        )
                        for k in b.params.cation_set
                    )
                else:
                    # This term does not exist for single cation systems
                    # However, need a valid result to calculate tau
                    return dG_dT_default
            elif i in b.params.anion_set and j in b.params.cation_set:
                # Eqn 43
                if len(b.params.anion_set) > 1:
                    return sum(
                        Y[k]
                        * _dG_dT_appr(
                            b, pobj, (j + ", " + i), (j + ", " + k), b.temperature
                        )
                        for k in b.params.anion_set
                    )
                else:
                    # This term does not exist for single anion systems
                    # However, need a valid result to calculate tau
                    return dG_dT_default
            elif (i in b.params.cation_set and j in b.params.cation_set) or (
                i in b.params.anion_set and j in b.params.anion_set
            ):
                # No like-ion interactions
                return Expression.Skip
            else:
                raise BurntToast(
                    "{} eNRTL model encountered unexpected component pair {}.".format(
                        b.name, (i, j)
                    )
                )

        b.add_component(
            pname + "_dG_dT",
            Expression(
                b.params.true_species_set,
                b.params.true_species_set,
                rule=rule_dG_dT_expr,
                doc="Temperature derivative of local interaction G term",
            ),
        )

        # Calculate tau terms
        def rule_tau_expr(b, i, j):
            if (pname, i) not in b.params.true_phase_component_set or (
                pname,
                j,
            ) not in b.params.true_phase_component_set:
                return Expression.Skip
            elif (i in molecular_set) and (j in molecular_set):
                # tau equal to parameter
                return tau_rule(b, pobj, i, j, b.temperature)
            elif (i in b.params.cation_set and j in b.params.cation_set) or (
                i in b.params.anion_set and j in b.params.anion_set
            ):
                # No like-ion interactions
                return Expression.Skip
            else:
                alpha = getattr(b, pname + "_alpha")
                G = getattr(b, pname + "_G")
                # Eqn 44
                return -log(G[i, j]) / alpha[i, j]

        b.add_component(
            pname + "_tau",
            Expression(
                b.params.true_species_set,
                b.params.true_species_set,
                rule=rule_tau_expr,
                doc="Binary interaction energy parameters",
            ),
        )
        def rule_dtau_dT_expr(b, i, j):
            if (pname, i) not in b.params.true_phase_component_set or (
                pname,
                j,
            ) not in b.params.true_phase_component_set:
                return Expression.Skip
            elif (i in molecular_set) and (j in molecular_set):
                # tau equal to parameter
                return dtau_dT_rule(b, pobj, i, j, b.temperature)
            elif (i in b.params.cation_set and j in b.params.cation_set) or (
                i in b.params.anion_set and j in b.params.anion_set
            ):
                # No like-ion interactions
                return Expression.Skip
            else:
                alpha = getattr(b, pname + "_alpha")
                dalpha_dT = getattr(b, pname + "_dalpha_dT")
                G = getattr(b, pname + "_G")
                dG_dT = getattr(b, pname + "_dG_dT")
                tau = getattr(b, pname + "_tau")
                # Eqn 44
                return -(
                    1 / (G[i, j] * alpha[i,j]) * dG_dT[i, j]
                    + tau[i, j] / alpha[i, j] * dalpha_dT[i, j]
                )

        b.add_component(
            pname + "_dtau_dT",
            Expression(
                b.params.true_species_set,
                b.params.true_species_set,
                rule=rule_dtau_dT_expr,
                doc="Temperature derivative of binary interaction energy parameters",
            ),
        )
        # Calculate reference state
        ref_state.ref_state(b, pname)

        # Long-range (PDH) contribution to activity coefficient
        def rule_log_gamma_pdh(b, j):
            A = getattr(b, pname + "_A_DH")
            Ix = getattr(b, pname + "_ionic_strength")
            I0 = getattr(b, pname + "_ionic_strength_ref")
            rho = ClosestApproach
            if j in molecular_set:
                # Eqn 69
                # Note typo in original paper. Correct power for I is (3/2)
                return 2 * A * Ix ** (3 / 2) / (1 + rho * Ix ** (1 / 2))
            elif j in b.params.ion_set:
                z = abs(cobj(b, j).config.charge)
                if pobj._reference_state_enrtl is Symmetric:
                    # Eqn 70
                    return -A * (
                        (2 * z**2 / rho)
                        * log((1 + rho * Ix**0.5) / (1 + rho * I0**0.5))
                        + (z**2 * Ix**0.5 - 2 * Ix ** (3 / 2)) / (1 + rho * Ix**0.5)
                        - (2 * Ix * I0**-0.5)
                        / (1 + rho * I0**0.5)
                        * ref_state.ndIdn(b, pname, j)
                    )
                else:
                    # Need to explicitly remove I0**-0.5 so it doesn't get evaluated for I0=0
                    return -A * (
                        (2 * z**2 / rho)
                        * log((1 + rho * Ix**0.5))
                        + (z**2 * Ix**0.5 - 2 * Ix ** (3 / 2)) / (1 + rho * Ix**0.5)
                    )
            else:
                raise BurntToast(
                    "{} eNRTL model encountered unexpected component.".format(b.name)
                )

        b.add_component(
            pname + "_log_gamma_pdh",
            Expression(
                b.params.true_species_set,
                rule=rule_log_gamma_pdh,
                doc="Long-range contribution to activity coefficient",
            ),
        )

        # Local contribution to activity coefficient
        def rule_log_gamma_lc_I(b, s):
            X = getattr(b, pname + "_X")
            G = getattr(b, pname + "_G")
            tau = getattr(b, pname + "_tau")

            return log_gamma_lc(b, pname, s, X, G, tau)

        b.add_component(
            pname + "_log_gamma_lc_I",
            Expression(
                b.params.true_species_set,
                rule=rule_log_gamma_lc_I,
                doc="Local contribution at actual state",
            ),
        )

        def rule_log_gamma_lc(b, s):
            log_gamma_lc_I = getattr(b, pname + "_log_gamma_lc_I")
            log_gamma_lc_I0 = getattr(b, pname + "_log_gamma_lc_I0")
            return log_gamma_lc_I[s] - log_gamma_lc_I0[s]

        b.add_component(
            pname + "_log_gamma_lc",
            Expression(
                b.params.true_species_set,
                rule=rule_log_gamma_lc,
                doc="Local contribution contribution to activity coefficient",
            ),
        )

        def rule_log_gamma_poynting(b, j):
            # TODO move to partial molar volume
            T = b.temperature
            v_comp = ENRTL.get_vol_mol_pure(b, "liq", j, T)
            # TODO do we know if having a vapor pressure is mutually
            # exclusive with being a Henry component?
            if (
                cobj(b, j).config.henry_component is not None
                and pname in cobj(b, j).config.henry_component
            ):
                P_star = b.params.pressure_ref
            elif cobj(b, j).config.has_vapor_pressure:
                P_star = get_method(b, "pressure_sat_comp", j)(
                    b, cobj(b, j), T
                )
            else:
                P_star = b.params.pressure_ref
            
            return (b.pressure - P_star) * v_comp / (ENRTL.gas_constant(b) * T)

        b.add_component(
            pname + "_log_gamma_poynting",
            Expression(
                b.params.true_species_set,
                rule=rule_log_gamma_poynting,
                doc="Poynting correction to activity coefficient",
            ),
        )

        # Overall log gamma
        def rule_log_gamma(b, j):
            pdh = getattr(b, pname + "_log_gamma_pdh")
            lc = getattr(b, pname + "_log_gamma_lc")
            born = getattr(b, pname + "_log_gamma_born")
            poynting = getattr(b, pname + "_log_gamma_poynting")
            return pdh[j] + lc[j] + born[j] + poynting[j]

        b.add_component(
            pname + "_log_gamma",
            Expression(
                b.params.true_species_set,
                rule=rule_log_gamma,
                doc="Log of activity coefficient",
            ),
        )

        # Activity coefficient of apparent species
        def rule_log_gamma_pm(b, j):
            cobj = b.params.get_component(j)

            if "dissociation_species" in cobj.config:
                dspec = cobj.config.dissociation_species

                n = 0
                d = 0
                for s in dspec:
                    dobj = b.params.get_component(s)
                    ln_g = getattr(b, pname + "_log_gamma")[s]
                    n += abs(dobj.config.charge) * ln_g
                    d += abs(dobj.config.charge)

                return n / d
            else:
                return getattr(b, pname + "_log_gamma")[j]

        b.add_component(
            pname + "_log_gamma_appr",
            Expression(
                b.params.apparent_species_set,
                rule=rule_log_gamma_pm,
                doc="Log of mean activity coefficient",
            ),
        )

    @staticmethod
    def calculate_scaling_factors(b, pobj):
        if not pobj.is_aqueous_phase():
            Ideal.calculate_scaling_factors(b, pobj)
        pass

    @staticmethod
    def act_phase_comp(b, p, j):
        return b.mole_frac_phase_comp[p, j] * b.act_coeff_phase_comp[p, j]

    @staticmethod
    def act_phase_comp_true(b, p, j):
        ln_gamma = getattr(b, p + "_log_gamma")
        return b.mole_frac_phase_comp_true[p, j] * exp(ln_gamma[j])

    @staticmethod
    def act_phase_comp_appr(b, p, j):
        ln_gamma = getattr(b, p + "_log_gamma_appr")
        return b.mole_frac_phase_comp_apparent[p, j] * exp(ln_gamma[j])
    
    @staticmethod
    def log_act_phase_comp(b, p, j):
        # TODO this should return the true value
        return b.log_mole_frac_phase_comp[p, j] + b.act_coeff_phase_comp[p, j]

    @staticmethod
    def log_act_phase_comp_true(b, p, j):
        ln_gamma = getattr(b, p + "_log_gamma")
        return b.log_mole_frac_phase_comp_true[p, j] + ln_gamma[j]

    @staticmethod
    def log_act_phase_comp_appr(b, p, j):
        ln_gamma = getattr(b, p + "_log_gamma_appr")
        return b.log_mole_frac_phase_comp_apparent[p, j] + ln_gamma[j]

    @staticmethod
    def act_coeff_phase_comp(b, p, j):
        if b.params.config.state_components == StateIndex.true:
            ln_gamma = getattr(b, p + "_log_gamma")
        else:
            ln_gamma = getattr(b, p + "_log_gamma_appr")
        return exp(ln_gamma[j])

    @staticmethod
    def act_coeff_phase_comp_true(b, p, j):
        ln_gamma = getattr(b, p + "_log_gamma")
        return exp(ln_gamma[j])

    @staticmethod
    def act_coeff_phase_comp_appr(b, p, j):
        ln_gamma = getattr(b, p + "_log_gamma_appr")
        return exp(ln_gamma[j])
    
    @staticmethod
    def log_act_coeff_phase_comp(b, p, j):
        if b.params.config.state_components == StateIndex.true:
            ln_gamma = getattr(b, p + "_log_gamma")
        else:
            ln_gamma = getattr(b, p + "_log_gamma_appr")
        return ln_gamma[j]

    @staticmethod
    def log_act_coeff_phase_comp_true(b, p, j):
        ln_gamma = getattr(b, p + "_log_gamma")
        return ln_gamma[j]

    @staticmethod
    def log_act_coeff_phase_comp_appr(b, p, j):
        ln_gamma = getattr(b, p + "_log_gamma_appr")
        return ln_gamma[j]

    @staticmethod
    def pressure_osm_phase(b, p):
        return (
            -ENRTL.gas_constant(b)
            * b.temperature
            * b.log_act_phase_solvents[p]
            / b.vol_mol_phase[p]
        )

    @staticmethod
    def vol_mol_phase(b, p):
        # eNRTL model uses apparent species for calculating molar volume
        # TODO : Need something more rigorus to handle concentrated solutions
        v_expr = 0
        for j in b.params.apparent_species_set:
            v_comp = ENRTL.get_vol_mol_pure(b, "liq", j, b.temperature)
            v_expr += b.mole_frac_phase_comp_apparent[p, j] * v_comp

        return v_expr
    
    # @staticmethod
    # def enth_mol_phase_comp_excess(b, p, j):
    #     if not hasattr(b, f"{p}_d_log_gamma_dT"):
    #         ENRTL._create_d_log_gamma_dT(b, p)
    #     d_log_gamma_dT = getattr(b, f"{p}_d_log_gamma_dT")
    #     return -ENRTL.gas_constant(b) * b.temperature ** 2 * d_log_gamma_dT[j]

    # These methods probably shouldn't exist    
    # @staticmethod
    # def enth_mol_phase_comp_ideal(b, p, j):
    #     return Ideal.enth_mol_phase_comp(b, p, j)
    
    # @staticmethod
    # def cp_mol_phase_comp_ideal(b, p, j):
    #     return Ideal.cp_mol_phase_comp(b, p, j)

    @staticmethod
    def cp_mol_phase(b, p):
        raise NotImplementedError(
            "Heat capacity deliberately left unimplemented due to complications with inherent reactions"
        )

    @staticmethod
    def cp_mol_phase_comp(b, p, j):
        raise NotImplementedError(
            "Heat capacity deliberately left unimplemented due to complications with inherent reactions"
        )
    
    @staticmethod
    def cv_mol_phase(b, p):
        raise NotImplementedError(
            "Heat capacity deliberately left unimplemented due to complications with inherent reactions"
        )

    @staticmethod
    def cv_mol_phase_comp(b, p, j):
        raise NotImplementedError(
            "Heat capacity deliberately left unimplemented due to complications with inherent reactions"
        )

    @staticmethod
    def enth_mol_phase_comp(b, p, j):
        # TODO we might be able to give a sensible answer to this via Perry (1981)
        pobj = b.params.get_phase(p)
        if not pobj.is_aqueous_phase():
            return Ideal.enth_mol_phase_comp(b, p, j)

        try:
            enth_mol_phase_basis = pobj.config.equation_of_state_options["enth_mol_phase_basis"]
        except KeyError:
            enth_mol_phase_basis = EnthMolPhaseBasis.true

        if enth_mol_phase_basis == EnthMolPhaseBasis.true:
            return Ideal.enth_mol_phase_comp(b, p, j) + ENRTL.enth_mol_phase_comp_excess(b, p, j)
        elif enth_mol_phase_basis == EnthMolPhaseBasis.apparent:
            raise AttributeError("Partial molar enthalpy is not well-defined when using apparent species as basis.")
        else:
            raise ConfigurationError

    @staticmethod
    def enth_mol_phase(b, p):
        pobj = b.params.get_phase(p)
        if not pobj.is_aqueous_phase():
            return Ideal.enth_mol_phase(b, p, j)
        # In case no inherent reactions are present
        try:
            inherent_rxn_idx = b.params.inherent_reaction_idx
        except AttributeError:
            enth_mol_ideal = sum(
                    b.mole_frac_phase_comp[p, j]
                    * Ideal.enth_mol_phase_comp(b, p, j)
                    for j in b.components_in_phase(p, true_basis=True)
                )
            return enth_mol_ideal + ENRTL.enth_mol_phase_excess(b, p)

        try:
            enth_mol_phase_basis = pobj.config.equation_of_state_options["enth_mol_phase_basis"]
        except KeyError:
            enth_mol_phase_basis = EnthMolPhaseBasis.true

        if enth_mol_phase_basis == EnthMolPhaseBasis.true:
            enth_mol_ideal = sum(
                    b.mole_frac_phase_comp_true[p, j]
                    * Ideal.enth_mol_phase_comp(b, p, j)
                    for j in b.components_in_phase(p, true_basis=True)
                )
            return enth_mol_ideal + ENRTL.enth_mol_phase_excess(b, p)
            
        elif enth_mol_phase_basis == EnthMolPhaseBasis.apparent:
            enth_mol_ideal = (
                sum(
                    b.mole_frac_phase_comp_apparent[p, j]
                    * Ideal.enth_mol_phase_comp(b, p, j)
                    for j in b.components_in_phase(p, true_basis=False)
                )
                + sum(
                    b.apparent_inherent_reaction_extent[r]
                    * b.dh_rxn[r]
                    # This flow_mol needs to be apparent for get_enthalpy_flow_terms
                    # to be correct. flow_mol_phase might be more correct, but
                    # right now inherent reactions are broken for multiphase systems
                    / b.flow_mol # TODO Fix later
                    for r in inherent_rxn_idx
                )
            )
        else:
            raise ConfigurationError
        # I think the PV term is included in the ideal contribution
        # enth_mol_ideal += (b.pressure - b.params.pressure_ref) / b.dens_mol_phase[p]
        flow_true = sum(
            b.flow_mol_phase_comp_true[p, j] 
            for j in b.components_in_phase(p, true_basis=True)
        )
        return enth_mol_ideal + flow_true / b.flow_mol * ENRTL.enth_mol_phase_excess(b, p)

    @staticmethod
    def energy_internal_mol_phase(b, p):
        pobj = b.params.get_phase(p)
        if not pobj.is_aqueous_phase():
            return Ideal.energy_internal_mol_phase(b, p)
        return  b.enth_mol_phase[p] - b.pressure / b.dens_mol_phase[p]

    @staticmethod
    def fug_phase_comp(b, p, j):
        T = b.temperature
        if (
            cobj(b, j).config.henry_component is not None
            and p in cobj(b, j).config.henry_component
        ):
            # Use Henry's Law
            return b.act_coeff_phase_comp_true[p, j] * henry_pressure(b, p, j, T)
        elif cobj(b, j).config.has_vapor_pressure:
            # Use Raoult's Law
            return (
                b.act_coeff_phase_comp_true[p, j]
                * b.mole_frac_phase_comp_true[p, j]
                * get_method(b, "pressure_sat_comp", j)(
                    b, cobj(b, j), T
                )
            )
        else:
            return Expression.Skip

    @staticmethod
    def fug_phase_comp_eq(b, p, j, pp):
        # Don't have a method to evaluate activity at other temperatures
        raise PropertyNotSupportedError(
            "Phase equilibrium calculations using eNRTL is not supported at this time."
        )
        

    @staticmethod
    def log_fug_phase_comp(b, p, j):
        pobj = b.params.get_phase(p)
        if not pobj.is_aqueous_phase():
            return Ideal.log_fug_phase_comp(b, p, j)

        pname = pobj.local_name
        log_gamma = getattr(b, f"{pname}_log_gamma")
        T = b.temperature
        if (
            cobj(b, j).config.henry_component is not None
            and p in cobj(b, j).config.henry_component
        ):
            # Use Henry's Law
            return log_gamma[j] + log_henry_pressure(b, p, j, T)
        elif cobj(b, j).config.has_vapor_pressure:
            # Use Raoult's Law
            return (
                log_gamma[j]
                + b.log_mole_frac_phase_comp_true[p, j]
                + b.log_pressure_sat_comp[j]
            )
        else:
            return Expression.Skip

    @staticmethod
    def log_fug_phase_comp_eq(b, p, j, pp):
        # Don't have a method to evaluate activity at other temperatures
        raise PropertyNotSupportedError(
            "Phase equilibrium calculations using eNRTL is not supported at this time."
            )


    # @staticmethod
    # def enth_mol_phase_excess(b, p):
    #     return sum(
    #         b.mole_frac_phase_comp_true[p, j]
    #         * ENRTL.enth_mol_phase_comp_excess(b, p, j)
    #         for j in b.components_in_phase(p, true_basis=True)
    #     )

    @staticmethod
    def enth_mol_phase_excess(b, p):
        pobj = b.params.get_phase(p)
        pname = pobj.local_name
        if not pobj.is_aqueous_phase():
            # TODO Fix this if/when we implement NRTL for organic phases
            units = b.params.get_metadata().derived_units
            return 0 * units.ENERGY_MOLE
        
        if not b.is_property_constructed(pname + "_d_log_gamma_lc_I0_dT"):
            raise NotImplementedError(
                "Enthalpy calculations are not implemented for your choice of reference state."
            )
        R = ENRTL.gas_constant(b)
        d_log_gamma_lc_I0_dT = getattr(b, pname + "_d_log_gamma_lc_I0_dT")

        X = getattr(b, pname + "_X")
        G = getattr(b, pname + "_G")
        dG_dT = getattr(b, pname + "_dG_dT")
        tau = getattr(b, pname + "_tau")
        dtau_dT = getattr(b, pname + "_dtau_dT")

        enth_mol_phase_excess_lc = -R * b.temperature**2 * (
            reduced_symmetric_gibbs_phase_temperature_derivative(
                b, X, G, dG_dT, tau, dtau_dT
            )
            - sum(
                b.mole_frac_phase_comp_true[p, s]
                * d_log_gamma_lc_I0_dT[s]
                for s in b.components_in_phase(p, true_basis=True)
            )
        )


        v = pyunits.convert(
            getattr(b, pname + "_vol_mol_solvent"), pyunits.m**3 / pyunits.mol
        )
        # TODO correct unit handling
        dv_dT = differentiate(
            v,
            b.temperature,
            mode=Modes.reverse_symbolic
        )
    
        eps = getattr(b, pname + "_relative_permittivity_solvent")
        d_eps_dT = getattr(b, pname + "_d_relative_permittivity_solvent_dT")
        A_DH = getattr(b, pname + "_A_DH")
        Ix = getattr(b, pname + "_ionic_strength")
        rho = ClosestApproach
        # Temperature derivative of Debye Huckel term
        dA_dT = -A_DH / 2 * (
            1/v * dv_dT + 3 * (1/eps) * d_eps_dT
        )

        # There's an additional term involving the derivative
        # of rho wrt temperature (see Pitzer and Simonson, 1986),
        # but since the main eNRTL treat rho as a constant,
        # we will here too
        enth_mol_phase_excess_DH = R * b.temperature**2 * (
            4 * Ix / rho * log(1 + rho * Ix**0.5) *dA_dT
        )

        enth_mol_phase_excess_born = getattr(
            b, pname + "_enth_mol_excess_born"
        )
        

        return (
            enth_mol_phase_excess_lc
            + enth_mol_phase_excess_DH
            + enth_mol_phase_excess_born
        )
    
    # @staticmethod
    # def _create_d_log_gamma_dT(b, p):
    #     pobj = b.params.get_phase(p)
    #     pname = pobj.local_name
    #     def rule_d_log_gamma_dT(b, j, pname):
    #         log_gamma = getattr(b, f"{pname}_log_gamma")
    #         return differentiate(
    #             expr=log_gamma[j],
    #             wrt=b.temperature,
    #             mode=Modes.reverse_symbolic
    #         )
    #     b.add_component(
    #         pname + "_d_log_gamma_dT",
    #         Expression(
    #             b.params.true_species_set,
    #             rule=partial(rule_d_log_gamma_dT, pname=pname),
    #             doc=f"Temperature derivative of {pname} activity coefficient",
    #         ),
    #     )






def log_gamma_lc(b, pname, s, X, G, tau):
    # General function for calculating local contributions
    # The same method can be used for both actual state and reference state
    # by providing different X, G and tau expressions.

    # Indices in expressions use same names as source paper
    # mp = m'
    # For the purposes of eNRTL, zwitterions are "molecules"
    molecular_set = b.params.solvent_set | b.params.solute_set | b.params.zwitterion_set
    aqu_species = b.params.true_species_set - b.params._non_aqueous_set

    if (pname, s) not in b.params.true_phase_component_set:
        # Non-aqueous component
        return Expression.Skip
    if s in b.params.cation_set:
        c = s
        Z = b.params.get_component(c).config.charge

        # Eqn 26
        return Z * (
            sum(
                (X[m] * G[c, m] / sum(X[i] * G[i, m] for i in aqu_species))
                * (
                    tau[c, m]
                    - (
                        sum(X[i] * G[i, m] * tau[i, m] for i in aqu_species)
                        / sum(X[i] * G[i, m] for i in aqu_species)
                    )
                )
                for m in molecular_set
            )
            + sum(
                X[i] * G[i, c] * tau[i, c] for i in (aqu_species - b.params.cation_set)
            )
            / sum(X[i] * G[i, c] for i in (aqu_species - b.params.cation_set))
            + sum(
                (
                    X[a]
                    * G[c, a]
                    / sum(X[i] * G[i, a] for i in (aqu_species - b.params.anion_set))
                )
                * (
                    tau[c, a]
                    - sum(
                        X[i] * G[i, a] * tau[i, a]
                        for i in (aqu_species - b.params.anion_set)
                    )
                    / sum(X[i] * G[i, a] for i in (aqu_species - b.params.anion_set))
                )
                for a in b.params.anion_set
            )
        )
    elif s in b.params.anion_set:
        a = s
        Z = abs(b.params.get_component(a).config.charge)

        # Eqn 27
        return Z * (
            sum(
                (X[m] * G[a, m] / sum(X[i] * G[i, m] for i in aqu_species))
                * (
                    tau[a, m]
                    - (
                        sum(X[i] * G[i, m] * tau[i, m] for i in aqu_species)
                        / sum(X[i] * G[i, m] for i in aqu_species)
                    )
                )
                for m in molecular_set
            )
            + sum(
                X[i] * G[i, a] * tau[i, a] for i in (aqu_species - b.params.anion_set)
            )
            / sum(X[i] * G[i, a] for i in (aqu_species - b.params.anion_set))
            + sum(
                (
                    X[c]
                    * G[a, c]
                    / sum(X[i] * G[i, c] for i in (aqu_species - b.params.cation_set))
                )
                * (
                    tau[a, c]
                    - sum(
                        X[i] * G[i, c] * tau[i, c]
                        for i in (aqu_species - b.params.cation_set)
                    )
                    / sum(X[i] * G[i, c] for i in (aqu_species - b.params.cation_set))
                )
                for c in b.params.cation_set
            )
        )
    else:
        m = s
        # Eqn 25
        return (
            sum(X[i] * G[i, m] * tau[i, m] for i in aqu_species)
            / sum(X[i] * G[i, m] for i in aqu_species)
            + sum(
                (X[mp] * G[m, mp] / sum(X[i] * G[i, mp] for i in aqu_species))
                * (
                    tau[m, mp]
                    - (
                        sum(X[i] * G[i, mp] * tau[i, mp] for i in aqu_species)
                        / sum(X[i] * G[i, mp] for i in aqu_species)
                    )
                )
                for mp in molecular_set
            )
            + sum(
                (
                    X[c]
                    * G[m, c]
                    / sum(X[i] * G[i, c] for i in (aqu_species - b.params.cation_set))
                )
                * (
                    tau[m, c]
                    - (
                        sum(
                            X[i] * G[i, c] * tau[i, c]
                            for i in (aqu_species - b.params.cation_set)
                        )
                        / sum(
                            X[i] * G[i, c] for i in (aqu_species - b.params.cation_set)
                        )
                    )
                )
                for c in b.params.cation_set
            )
            + sum(
                (
                    X[a]
                    * G[m, a]
                    / sum(X[i] * G[i, a] for i in (aqu_species - b.params.anion_set))
                )
                * (
                    tau[m, a]
                    - (
                        sum(
                            X[i] * G[i, a] * tau[i, a]
                            for i in (aqu_species - b.params.anion_set)
                        )
                        / sum(
                            X[i] * G[i, a] for i in (aqu_species - b.params.anion_set)
                        )
                    )
                )
                for a in b.params.anion_set
            )
        )
    
def reduced_symmetric_gibbs_phase(b, X, G, tau):
    molecular_set = b.params.solvent_set | b.params.solute_set | b.params.zwitterion_set
    aqu_species = b.params.true_species_set - b.params._non_aqueous_set

    #  Eqn. 20
    return sum(
        X[m] * sum(X[i] * G[i, m] * tau[i, m] for i in aqu_species)
            / sum(X[i] * G[i, m] for i in aqu_species)
        for m in molecular_set
    ) + sum(
        X[a]* sum(
            X[i] * G[i, a] * tau[i, a] for i in (aqu_species - b.params.anion_set)
        ) / sum(X[i] * G[i, a] for i in (aqu_species - b.params.anion_set))
        for a in b.params.anion_set
    ) + sum(
        X[c] * sum(
                X[i] * G[i, c] * tau[i, c] for i in (aqu_species - b.params.cation_set)
            ) / sum(X[i] * G[i, c] for i in (aqu_species - b.params.cation_set))
        for c in b.params.cation_set
    )

def reduced_symmetric_gibbs_phase_temperature_derivative(b, X, G, dG_dT, tau, dtau_dT):
    # Not dimensionless, units 1/T
    molecular_set = b.params.solvent_set | b.params.solute_set | b.params.zwitterion_set
    aqu_species = b.params.true_species_set - b.params._non_aqueous_set

    Gamma_m = sum(
        X[m] * (
            sum(X[i] * (dG_dT[i, m] * tau[i, m] + G[i, m] * dtau_dT[i, m]) for i in aqu_species)
            / sum(X[i] * G[i, m] for i in aqu_species)
            - sum(X[i] * G[i, m] * tau[i, m] for i in aqu_species)
            * sum(X[i] * dG_dT[i, m] for i in aqu_species)
            / sum(X[i] * G[i, m] for i in aqu_species) ** 2
        )
        for m in molecular_set
    )
    Gamma_c = sum(
        X[c] * (
            sum(
                X[i] * (dG_dT[i, c] * tau[i, c] + G[i, c] * dtau_dT[i, c] )
                for i in (aqu_species - b.params.cation_set)
            ) / sum(X[i] * G[i, c] for i in (aqu_species - b.params.cation_set))
            - sum(
                X[i] * G[i, c] * tau[i, c] for i in (aqu_species - b.params.cation_set)
            ) * sum(X[i] * dG_dT[i, c] for i in (aqu_species - b.params.cation_set)) 
            / sum(X[i] * G[i, c] for i in (aqu_species - b.params.cation_set)) ** 2
        )
        for c in b.params.cation_set
    )
    Gamma_a = sum(
        X[a]* (
            sum(
                X[i] * (dG_dT[i, a] * tau[i, a] + G[i, a] * dtau_dT[i, a])
                for i in (aqu_species - b.params.anion_set)
            ) / sum(X[i] * G[i, a] for i in (aqu_species - b.params.anion_set))
            - sum(
                X[i] * G[i, a] * tau[i, a] for i in (aqu_species - b.params.anion_set)
            ) * sum(X[i] * dG_dT[i, a] for i in (aqu_species - b.params.anion_set))
            / sum(X[i] * G[i, a] for i in (aqu_species - b.params.anion_set)) ** 2
        )
        for a in b.params.anion_set
    )

    return  Gamma_m + Gamma_c + Gamma_a
