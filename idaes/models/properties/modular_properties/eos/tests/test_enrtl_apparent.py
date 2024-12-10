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
Tests for eNRTL methods

Author: Andrew Lee
"""
from copy import deepcopy

import pytest

import sympy as sy

from pyomo.environ import (
    ConcreteModel,
    Expression,
    exp,
    log,
    Set,
    units as pyunits,
    value,
    Var,
)
from pyomo.util.check_units import assert_units_equivalent
from pyomo.core.expr.sympy_tools import sympyify_expression

from idaes.core import AqueousPhase, Solvent, Solute, Apparent, Anion, Cation
from idaes.core.util.constants import Constants
from idaes.models.properties.modular_properties.eos.enrtl import ENRTL, EnthMolPhaseBasis
from idaes.models.properties.modular_properties.eos.enrtl_reference_states import InfiniteDilutionSingleSolvent
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
    StateIndex,
)
from idaes.models.properties.modular_properties.state_definitions import FTPx
from idaes.models.properties.modular_properties.pure.electrolyte import (
    relative_permittivity_constant,
)
from idaes.models.properties.modular_properties.base.utility import ConcentrationForm
from idaes.models.properties.modular_properties.reactions.equilibrium_forms import log_power_law_equil
from idaes.models.properties.modular_properties.reactions.equilibrium_constant import ConstantKeq
from idaes.models.properties.modular_properties.reactions.dh_rxn import constant_dh_rxn
from idaes.models.properties.modular_properties.eos.enrtl_reference_states import Symmetric, InfiniteDilutionSingleSolvent, Unsymmetric
from idaes.models.properties.modular_properties.eos.enrtl_parameters import TwoParameterTau
from idaes.core.util.exceptions import ConfigurationError
import idaes.logger as idaeslog


def dummy_method(b, *args, **kwargs):
    return 42.0 * pyunits.mol / pyunits.m**3


configuration = {
    "components": {
        "H2O": {
            "type": Solvent,
            "dens_mol_liq_comp": dummy_method,
            "relative_permittivity_liq_comp": relative_permittivity_constant,
            "parameter_data": {
                "mw": (18e-3, pyunits.kg / pyunits.mol),
                "relative_permittivity_liq_comp": 101,
            },
        },
        "C6H12": {
            "type": Solvent,
            "dens_mol_liq_comp": dummy_method,
            "relative_permittivity_liq_comp": relative_permittivity_constant,
            "parameter_data": {
                "mw": (84e-3, pyunits.kg / pyunits.mol),
                "relative_permittivity_liq_comp": 102,
            },
        },
        "NaCl": {"type": Solute},
        "HCl": {"type": Solute},
        "NaOH": {"type": Solute},
        "Na+": {"type": Cation, "charge": +1, "parameter_data": {"born_radius": (1.0, pyunits.angstrom)}},
        "H+": {"type": Cation, "charge": +1, "parameter_data": {"born_radius": (2.0, pyunits.angstrom)}},
        "Cl-": {"type": Anion, "charge": -1, "parameter_data": {"born_radius": (3.0, pyunits.angstrom)}},
        "OH-": {"type": Anion, "charge": -1, "parameter_data": {"born_radius": (4.0, pyunits.angstrom)}},
    },
    "phases": {
        "Liq": {
            "type": AqueousPhase,
            "equation_of_state": ENRTL,
            "equation_of_state_options": {
                "property_basis": "true",
                "tau_rule": TwoParameterTau,
                "reference_state": InfiniteDilutionSingleSolvent,
                "reference_component": "H2O",
                "enth_mol_phase_basis": EnthMolPhaseBasis.apparent,
            },
        }
    },
    "base_units": {
        "time": pyunits.s,
        "length": pyunits.m,
        "mass": pyunits.kg,
        "amount": pyunits.mol,
        "temperature": pyunits.K,
    },
    "state_definition": FTPx,
    "state_components": StateIndex.apparent,
    "pressure_ref": 1e5,
    "temperature_ref": 300,
    "inherent_reactions": {
        "NaCl_dissociation": {
            "stoichiometry": {
                ("Liq", "NaCl"): -1,
                ("Liq", "Na+"): 1,
                ("Liq", "Cl-"): 1,
            },
            "equilibrium_constant": ConstantKeq,
            "heat_of_reaction": constant_dh_rxn,
            "equilibrium_form": log_power_law_equil,
            "concentration_form": ConcentrationForm.activity,
            "parameter_data":{
                "k_eq_ref": (3.14, pyunits.dimensionless),
                "dh_rxn_ref": (2.71, pyunits.kJ/pyunits.mol)
            },
        },
        "HCl_dissociation": {
            "stoichiometry": {
                ("Liq", "HCl"): -1,
                ("Liq", "H+"): 1,
                ("Liq", "Cl-"): 1,
            },
            "equilibrium_constant": ConstantKeq,
            "heat_of_reaction": constant_dh_rxn,
            "equilibrium_form": log_power_law_equil,
            "concentration_form": ConcentrationForm.activity,
            "parameter_data":{
                "k_eq_ref": (3.14, pyunits.dimensionless),
                "dh_rxn_ref": (2.71, pyunits.kJ/pyunits.mol)
            },
        },
        "NaOH_dissociation": {
            #TODO
            "stoichiometry": {
                ("Liq", "NaOH"): -1,
                ("Liq", "Na+"): 1,
                ("Liq", "OH-"): 1,
            },
            "equilibrium_constant": ConstantKeq,
            "heat_of_reaction": constant_dh_rxn,
            "equilibrium_form": log_power_law_equil,
            "concentration_form": ConcentrationForm.activity,
            "parameter_data":{
                "k_eq_ref": (3.14, pyunits.dimensionless),
                "dh_rxn_ref": (2.71, pyunits.kJ/pyunits.mol)
            },
        },
    },
    "parameter_data":{
        "Liq_alpha": {
            # Molecule-molecule interactions have default value
            # of 0.3, all others have default value of 0.2
            ("H2O", "C6H12"): 0.1, 
            ("Na+, OH-", "H2O"): 0.4,
        },
        "Liq_tau_A": {
            # Default value of 0
            ("C6H12", "H2O"): 5,

            ("H2O", "H+, Cl-"): -3,
            ("C6H12", "Na+, Cl-"): 8,
        },
        "Liq_tau_B": {
            # Default value of 0
            ("NaCl","H2O"): -5,
            ("H+, OH-", "C6H12"): 7,
        },
    }
}

_all_components_set = ["H2O", "C6H12", "NaCl", "HCl", "NaOH", "Na+", "Cl-", "H+", "OH-"]
_solvent_set = ["H2O", "C6H12"]
_solute_set =  ["NaCl", "HCl", "NaOH"]
_uncharged_components_set = ["H2O", "C6H12", "NaCl", "HCl", "NaOH"]
_cation_set = ["Na+", "H+"]
_anion_set = ["Cl-", "OH-"]
_charged_components_set = _cation_set + _anion_set

def _test_common(model):
    assert isinstance(model.state[1].Liq_X, Expression)
    assert len(model.state[1].Liq_X) == 9
    for j in model.state[1].Liq_X:
        if j in _uncharged_components_set:
            # _X should be mole_frac_phase_comp_true
            assert str(model.state[1].Liq_X[j].expr) == str(
                model.state[1].mole_frac_phase_comp_true["Liq", j]
            )
        else:
            # _X should be mutiplied by |charge|
            assert str(model.state[1].Liq_X[j].expr) == str(
                model.state[1].mole_frac_phase_comp_true["Liq", j]
                * abs(model.params.get_component(j).config.charge)
            )

    assert isinstance(model.state[1].Liq_Y, Expression)
    assert len(model.state[1].Liq_Y) == 4
    for j in model.state[1].Liq_Y:
        if j in ["H+", "Na+"]:
            assert str(model.state[1].Liq_Y[j].expr) == str(
                model.state[1].Liq_X[j]
                / (model.state[1].Liq_X["Na+"] + model.state[1].Liq_X["H+"])
            )
        else:
            assert str(model.state[1].Liq_Y[j].expr) == str(
                model.state[1].Liq_X[j]
                / (model.state[1].Liq_X["Cl-"] + model.state[1].Liq_X["OH-"])
            )

    assert isinstance(model.state[1].Liq_ionic_strength, Expression)
    assert len(model.state[1].Liq_ionic_strength) == 1
    assert str(model.state[1].Liq_ionic_strength.expr) == str(
        0.5
        * (
            model.params.get_component("Cl-").config.charge ** 2
            * model.state[1].mole_frac_phase_comp_true["Liq", "Cl-"]
            + model.params.get_component("OH-").config.charge ** 2
            * model.state[1].mole_frac_phase_comp_true["Liq", "OH-"]
            + model.params.get_component("Na+").config.charge ** 2
            * model.state[1].mole_frac_phase_comp_true["Liq", "Na+"]
            + model.params.get_component("H+").config.charge ** 2
            * model.state[1].mole_frac_phase_comp_true["Liq", "H+"]
        )
    )

    assert isinstance(model.state[1].Liq_A_DH, Expression)
    assert len(model.state[1].Liq_A_DH) == 1
    assert_units_equivalent(model.state[1].Liq_A_DH, pyunits.dimensionless)
    assert str(model.state[1].Liq_A_DH.expr) == str(
        (1 / 3)
        * (
            2
            * Constants.pi
            * Constants.avogadro_number
            / model.state[1].Liq_vol_mol_solvent
        )
        ** 0.5
        * (
            Constants.elemental_charge**2
            / (
                4
                * Constants.pi
                * model.state[1].Liq_relative_permittivity_solvent
                * Constants.vacuum_electric_permittivity
                * Constants.boltzmann_constant
                * model.state[1].temperature
            )
        )
        ** (3 / 2)
    )

    assert isinstance(model.state[1].Liq_log_gamma_pdh, Expression)
    assert len(model.state[1].Liq_log_gamma_pdh) == 9
    A = model.state[1].Liq_A_DH
    Ix = model.state[1].Liq_ionic_strength
    I0 = model.state[1].Liq_ionic_strength_ref
    rho = 14.9
    ref_state = model.params.Liq._reference_state_enrtl
    for j in model.state[1].Liq_log_gamma_pdh:
        assert j in _all_components_set
        if j in _uncharged_components_set:
            assert str(model.state[1].Liq_log_gamma_pdh[j].expr) == str(
                2
                * model.state[1].Liq_A_DH
                * model.state[1].Liq_ionic_strength ** (3 / 2)
                / (1 + 14.9 * model.state[1].Liq_ionic_strength ** (1 / 2))
            )
        else:
            z = model.params.get_component(j).config.charge
            if ref_state is Symmetric:
                assert str(model.state[1].Liq_log_gamma_pdh[j].expr) == str(
                    -A * (
                        (2 * z**2 / rho)
                        * log((1 + rho * Ix**0.5) / (1 + rho * I0**0.5))
                        + (z**2 * Ix**0.5 - 2*Ix**1.5) / (1 + rho * Ix**0.5)
                        - (2 * Ix * I0**-0.5)
                        / (1 + rho * I0**0.5)
                        * ref_state.ndIdn(model.state[1], "Liq", j) 
                    )
                )
            elif (
                ref_state is InfiniteDilutionSingleSolvent
                or ref_state is Unsymmetric
            ):
                assert str(model.state[1].Liq_log_gamma_pdh[j].expr) == str(
                    -A * (
                        (2 * z**2 / rho)
                        * log((1 + rho * Ix**0.5))
                        + (z**2 * Ix**0.5 - 2*Ix**1.5) / (1 + rho * Ix**0.5)
                    )
                )
            else:
                # Invalid reference state
                assert False
            


    assert isinstance(model.state[1].Liq_log_gamma_lc_I, Expression)
    assert len(model.state[1].Liq_log_gamma_lc_I) == 9
    for k in model.state[1].Liq_log_gamma_lc_I:
        assert k in _all_components_set

    assert isinstance(model.state[1].Liq_log_gamma_lc, Expression)
    assert len(model.state[1].Liq_log_gamma_lc) == 9
    for k in model.state[1].Liq_log_gamma_lc:
        assert k in _all_components_set
        assert str(model.state[1].Liq_log_gamma_lc[k].expr) == str(
            model.state[1].Liq_log_gamma_lc_I[k]
            - model.state[1].Liq_log_gamma_lc_I0[k]
        )

    assert isinstance(model.state[1].Liq_log_gamma, Expression)
    assert len(model.state[1].Liq_log_gamma) == 9
    for k, v in model.state[1].Liq_log_gamma.items():
        assert str(model.state[1].Liq_log_gamma[k].expr) == str(
            model.state[1].Liq_log_gamma_pdh[k]
            + model.state[1].Liq_log_gamma_lc[k]
            + model.state[1].Liq_log_gamma_born[k]
            + model.state[1].Liq_log_gamma_poynting[k]
        )
    # assert isinstance(model.state[1].enth_mol_phase, Expression)
    # assert len(model.state[1].enth_mol_phase) == 1
    # for k in model.state[1].enth_mol_phase:
    #     assert k in {"Liq"}
    #     enth_mol_ideal = sum(
    #         model.state[1].mole_frac_phase_comp_true["Liq", j]
    #         * model.state[1].enth_mol_phase_comp["Liq", j]
    #         for j in model.state[1].components_in_phase("Liq", true_basis=True)
    #     )
    #     enth_mol_excess = (

    #     )
    #     assert enth_mol_ideal

def _test_constant_alpha(model):
    assert isinstance(model.state[1].Liq_alpha, Expression)
    assert len(model.state[1].Liq_alpha) == len(_all_components_set)**2 - len(_cation_set)**2 - len(_anion_set)**2
    assert len(model.state[1].Liq_dalpha_dT) == len(_all_components_set)**2 - len(_cation_set)**2 - len(_anion_set)**2

    # Molecule-molecule interactions
    for i in _uncharged_components_set:
        for j in _uncharged_components_set:
            if (i, j) in model.params.Liq.alpha:
                alpha_param = model.params.Liq.alpha[i, j]
            elif (j, i) in model.params.Liq.alpha:
                alpha_param = model.params.Liq.alpha[j, i]
            else:
                raise AssertionError
            assert str(model.state[1].Liq_alpha[i, j].expr) == str(alpha_param)
            assert str(model.state[1].Liq_dalpha_dT[i, j].expr) == str(0/pyunits.K)
    # Molecule-ion interactions
    for mol in _uncharged_components_set:
        for ion in _charged_components_set:
            alpha = 0
            dalpha_dT = 0
            if ion in _cation_set:
                anion = _anion_set[0]
                for anion in _anion_set:
                    alpha += model.state[1].Liq_Y[anion] * model.params.Liq.alpha[mol, f"{ion}, {anion}"]
                    dalpha_dT += model.state[1].Liq_Y[anion] * (0 / pyunits.K)
            elif ion in _anion_set:
                cation = _cation_set[0]
                for cation in _cation_set:
                    alpha += model.state[1].Liq_Y[cation] * model.params.Liq.alpha[mol, f"{cation}, {ion}"]
                    dalpha_dT += model.state[1].Liq_Y[cation] * (0 / pyunits.K)
            else:
                # This branch should be impossible
                raise AssertionError("Getting here should be impossible.")
            assert str(model.state[1].Liq_alpha[mol, ion].expr) == str(alpha)
            assert str(model.state[1].Liq_alpha[ion, mol].expr) == str(alpha)
            assert str(model.state[1].Liq_dalpha_dT[mol, ion].expr) == str(dalpha_dT)
            assert str(model.state[1].Liq_dalpha_dT[ion, mol].expr) == str(dalpha_dT)
 
    for ion1 in _charged_components_set:
        for ion2 in _charged_components_set:
            if (
                (ion1 in _cation_set and ion2 in _cation_set)
                or (ion1 in _anion_set and ion2 in _anion_set)
            ):
                # Like charge interactions don't exist
                assert (ion1, ion2) not in model.state[1].Liq_alpha
            else:
                alpha = 0
                dalpha_dT = 0
                if ion1 in _cation_set:
                    for cation in _cation_set:
                        if cation == ion1:
                            alpha += model.state[1].Liq_Y[ion1] * 0.2
                            dalpha_dT += model.state[1].Liq_Y[ion1] * (0 / pyunits.K) 
                        else:
                            ion_pair_pair = (f"{ion1}, {ion2}", f"{cation}, {ion2}")
                            if ion_pair_pair in model.params.Liq.alpha:
                                param_alpha = model.params.Liq.alpha[ion_pair_pair]
                            elif ion_pair_pair[::-1] in model.params.Liq.alpha:
                                param_alpha = model.params.Liq.alpha[ion_pair_pair[::-1]]
                            else:
                                raise AssertionError
                            alpha += model.state[1].Liq_Y[cation] * param_alpha
                            dalpha_dT += model.state[1].Liq_Y[cation] * (0 / pyunits.K) 
                    assert str(model.state[1].Liq_alpha[ion1, ion2].expr) == str(alpha)
                elif ion1 in _anion_set:
                    for anion in _anion_set:
                        if anion == ion1:
                            alpha += model.state[1].Liq_Y[ion1] * 0.2
                            dalpha_dT += model.state[1].Liq_Y[ion1] * (0 / pyunits.K)
                        else:
                            ion_pair_pair = (f"{ion2}, {ion1}", f"{ion2}, {anion}")
                            if ion_pair_pair in model.params.Liq.alpha:
                                param_alpha = model.params.Liq.alpha[ion_pair_pair]
                            elif ion_pair_pair[::-1] in model.params.Liq.alpha:
                                param_alpha = model.params.Liq.alpha[ion_pair_pair[::-1]]
                            else:
                                raise AssertionError
                            alpha += model.state[1].Liq_Y[anion] * param_alpha
                            dalpha_dT += model.state[1].Liq_Y[anion] * (0 / pyunits.K)
                    assert str(model.state[1].Liq_alpha[ion1, ion2].expr) == str(alpha)
                    assert str(model.state[1].Liq_dalpha_dT[ion1, ion2].expr) == str(dalpha_dT)
                else:
                    raise AssertionError("Getting here should be impossible.")

def _test_G_constant_alpha_two_parameter_tau(model):
    assert isinstance(model.state[1].Liq_G, Expression)
    assert len(model.state[1].Liq_G) == len(_all_components_set)**2 - len(_cation_set)**2 - len(_anion_set)**2

    # Molecule-molecule interactions
    for i in _uncharged_components_set:
        for j in _uncharged_components_set:
            if i == j:
                assert str(model.state[1].Liq_G[i, j].expr) == str(1.0)
            else:
                if (i, j) in model.params.Liq.alpha:
                    alpha = model.params.Liq.alpha[i, j]
                else:
                    alpha = model.params.Liq.alpha[j, i]
                assert str(model.state[1].Liq_G[i, j].expr) == str(
                    exp(
                        -alpha
                        * (
                            model.params.Liq.tau_A[i, j]
                            + model.params.Liq.tau_B[i, j] / model.state[1].temperature
                        )
                    )
                )
                assert str(model.state[1].Liq_dG_dT[i, j].expr) == str(
                    - exp(
                        -alpha
                        * (
                            model.params.Liq.tau_A[i, j]
                            + model.params.Liq.tau_B[i, j] / model.state[1].temperature
                        )
                    ) * (
                        0 / pyunits.K * (
                            model.params.Liq.tau_A[i, j]
                            + model.params.Liq.tau_B[i, j] / model.state[1].temperature
                        )
                        + alpha * (
                            -model.params.Liq.tau_B[i, j] / (model.state[1].temperature**2)
                        )
                    )
                )

    # Molecule-ion interactions
    for mol in _uncharged_components_set:
        for ion in _charged_components_set:
            G_mol_ion = 0
            G_ion_mol = 0
            dG_mol_ion_dT = 0
            dG_ion_mol_dT = 0
            if ion in _cation_set:
                for anion in _anion_set:
                    ion_pair = f"{ion}, {anion}"
                    alpha = model.params.Liq.alpha[mol, ion_pair]
                    G_mol_ion += model.state[1].Liq_Y[anion] * exp(
                        - alpha
                        * (
                            model.params.Liq.tau_A[mol, ion_pair]
                            + model.params.Liq.tau_B[mol, ion_pair] / model.state[1].temperature
                        )
                    )

                    dG_mol_ion_dT += model.state[1].Liq_Y[anion] * - exp(
                        - alpha
                        * (
                            model.params.Liq.tau_A[mol, ion_pair]
                            + model.params.Liq.tau_B[mol, ion_pair] / model.state[1].temperature
                        )
                    ) * (
                        0 / pyunits.K * (
                            model.params.Liq.tau_A[mol, ion_pair]
                            + model.params.Liq.tau_B[mol, ion_pair] / model.state[1].temperature
                        )
                        + alpha * (
                            -model.params.Liq.tau_B[mol, ion_pair] / (model.state[1].temperature**2)
                        )
                    )

                    G_ion_mol += model.state[1].Liq_Y[anion] * exp(
                        - alpha
                        * (
                            model.params.Liq.tau_A[ion_pair, mol]
                            + model.params.Liq.tau_B[ion_pair, mol] / model.state[1].temperature
                        )
                    )

                    dG_ion_mol_dT += model.state[1].Liq_Y[anion] * - exp(
                        - alpha
                        * (
                            model.params.Liq.tau_A[ion_pair, mol]
                            + model.params.Liq.tau_B[ion_pair, mol] / model.state[1].temperature
                        )
                    ) * (
                        0 / pyunits.K * (
                            model.params.Liq.tau_A[ion_pair, mol]
                            + model.params.Liq.tau_B[ion_pair, mol] / model.state[1].temperature
                        )
                        + alpha * (
                            -model.params.Liq.tau_B[ion_pair,mol] / (model.state[1].temperature**2)
                        )
                    )
            elif ion in _anion_set:
                for cation in _cation_set:
                    ion_pair = f"{cation}, {ion}"
                    alpha = model.params.Liq.alpha[mol, ion_pair]
                    G_mol_ion += model.state[1].Liq_Y[cation] * exp(
                        -alpha
                        * (
                            model.params.Liq.tau_A[mol, ion_pair]
                            + model.params.Liq.tau_B[mol, ion_pair] / model.state[1].temperature
                        )
                    )

                    dG_mol_ion_dT += model.state[1].Liq_Y[cation] * - exp(
                        - alpha
                        * (
                            model.params.Liq.tau_A[mol, ion_pair]
                            + model.params.Liq.tau_B[mol, ion_pair] / model.state[1].temperature
                        )
                    ) * (
                        0 / pyunits.K * (
                            model.params.Liq.tau_A[mol, ion_pair]
                            + model.params.Liq.tau_B[mol, ion_pair] / model.state[1].temperature
                        )
                        + alpha * (
                            -model.params.Liq.tau_B[mol, ion_pair] / (model.state[1].temperature**2)
                        )
                    )

                    G_ion_mol += model.state[1].Liq_Y[cation] * exp(
                        -alpha
                        * (
                            model.params.Liq.tau_A[ion_pair, mol]
                            + model.params.Liq.tau_B[ion_pair, mol] / model.state[1].temperature
                        )
                    )
                    dG_ion_mol_dT += model.state[1].Liq_Y[cation] * - exp(
                        - alpha
                        * (
                            model.params.Liq.tau_A[ion_pair, mol]
                            + model.params.Liq.tau_B[ion_pair, mol] / model.state[1].temperature
                        )
                    ) * (
                        0 / pyunits.K * (
                            model.params.Liq.tau_A[ion_pair, mol]
                            + model.params.Liq.tau_B[ion_pair, mol] / model.state[1].temperature
                        )
                        + alpha * (
                            -model.params.Liq.tau_B[ion_pair,mol] / (model.state[1].temperature**2)
                        )
                    )
            else:
                # This branch should be impossible
                raise AssertionError("Getting here should be impossible.")
            

            assert str(model.state[1].Liq_G[mol, ion].expr) == str(G_mol_ion)
            assert str(model.state[1].Liq_G[ion, mol].expr) == str(G_ion_mol)
            _, sympy_expr = sympyify_expression(model.state[1].Liq_dG_dT[mol, ion].expr - dG_mol_ion_dT)
            assert sy.simplify(sympy_expr) == 0
            # assert str(model.state[1].Liq_dG_dT[mol, ion].expr) == str(dG_mol_ion_dT)
            _, sympy_expr = sympyify_expression(model.state[1].Liq_dG_dT[ion, mol].expr - dG_ion_mol_dT)
            assert sy.simplify(sympy_expr) == 0
            # assert str(model.state[1].Liq_dG_dT[ion, mol].expr) == str(dG_ion_mol_dT)

    # Ion-ion interactions

    for ion1 in _charged_components_set:
        for ion2 in _charged_components_set:
            if (
                (ion1 in _cation_set and ion2 in _cation_set)
                or (ion1 in _anion_set and ion2 in _anion_set)
            ):
                # Like charge interactions don't exist
                assert (ion1, ion2) not in model.state[1].Liq_G
            else:
                G_ion1_ion2 = 0
                dG_ion1_ion2_dT = 0
                if ion1 in _cation_set:
                    for cation in _cation_set:
                        if cation == ion1:
                            G_ion1_ion2 += model.state[1].Liq_Y[ion1]
                            dG_ion1_ion2_dT += model.state[1].Liq_Y[ion1] * model.state[1].Liq_dG_dT_default
                        else:
                            ion_pair_pair = (f"{ion1}, {ion2}", f"{cation}, {ion2}")
                            if ion_pair_pair in model.params.Liq.alpha:
                                param_alpha = model.params.Liq.alpha[ion_pair_pair]
                            elif ion_pair_pair[::-1] in model.params.Liq.alpha:
                                param_alpha = model.params.Liq.alpha[ion_pair_pair[::-1]]
                            else:
                                raise AssertionError
                            G_ion1_ion2 += model.state[1].Liq_Y[cation] * exp(
                                -param_alpha * (
                                    model.params.Liq.tau_A[ion_pair_pair]
                                    + model.params.Liq.tau_B[ion_pair_pair] / model.state[1].temperature
                                )
                            )
                            dG_ion1_ion2_dT += model.state[1].Liq_Y[cation] * -exp(
                                -param_alpha * (
                                    model.params.Liq.tau_A[ion_pair_pair]
                                    + model.params.Liq.tau_B[ion_pair_pair] / model.state[1].temperature
                                )
                            ) * (
                                0 / pyunits.K * (
                                    model.params.Liq.tau_A[ion_pair_pair]
                                    + model.params.Liq.tau_B[ion_pair_pair] / model.state[1].temperature
                                )
                                + param_alpha * (
                                    -model.params.Liq.tau_B[ion_pair_pair] / (model.state[1].temperature**2)
                                )
                            )
                elif ion1 in _anion_set:
                    for anion in _anion_set:
                        if anion == ion1:
                            G_ion1_ion2 += model.state[1].Liq_Y[ion1]
                            dG_ion1_ion2_dT += model.state[1].Liq_Y[ion1] * model.state[1].Liq_dG_dT_default
                        else:
                            ion_pair_pair = (f"{ion2}, {ion1}", f"{ion2}, {anion}")
                            if ion_pair_pair in model.params.Liq.alpha:
                                param_alpha = model.params.Liq.alpha[ion_pair_pair]
                            elif ion_pair_pair[::-1] in model.params.Liq.alpha:
                                param_alpha = model.params.Liq.alpha[ion_pair_pair[::-1]]
                            else:
                                raise AssertionError
                            G_ion1_ion2 += model.state[1].Liq_Y[anion] * exp(
                                -param_alpha * (
                                    model.params.Liq.tau_A[ion_pair_pair]
                                    + model.params.Liq.tau_B[ion_pair_pair] / model.state[1].temperature
                                )
                            )
                            dG_ion1_ion2_dT += model.state[1].Liq_Y[anion] * -exp(
                                -param_alpha * (
                                    model.params.Liq.tau_A[ion_pair_pair]
                                    + model.params.Liq.tau_B[ion_pair_pair] / model.state[1].temperature
                                )
                            ) * (
                                0 / pyunits.K * (
                                    model.params.Liq.tau_A[ion_pair_pair]
                                    + model.params.Liq.tau_B[ion_pair_pair] / model.state[1].temperature
                                )
                                + param_alpha * (
                                    -model.params.Liq.tau_B[ion_pair_pair] / (model.state[1].temperature**2)
                                )
                            )
                else:
                    raise AssertionError("Getting here should be impossible.")
                _, sympy_expr = sympyify_expression(model.state[1].Liq_G[ion1, ion2].expr - G_ion1_ion2)
                assert sy.simplify(sympy_expr) == 0
                _, sympy_expr = sympyify_expression(model.state[1].Liq_dG_dT[ion1, ion2].expr - dG_ion1_ion2_dT)
                assert sy.simplify(sympy_expr) == 0

def _test_two_parameter_tau(model):
    assert isinstance(model.state[1].Liq_tau, Expression)
    assert len(model.state[1].Liq_tau) == len(_all_components_set)**2 - len(_cation_set)**2 - len(_anion_set)**2

    # Molecule-molecule interactions
    for i in _uncharged_components_set:
        for j in _uncharged_components_set:
            assert str(model.state[1].Liq_tau[i, j].expr) == str(
                model.params.Liq.tau_A[i, j] + model.params.Liq.tau_B[i, j] / model.state[1].temperature
            )
            assert str(model.state[1].Liq_dtau_dT[i, j].expr) == str(
                -model.params.Liq.tau_B[i, j] / (model.state[1].temperature**2)
            )

    assert False

    for i, j in model.state[1].Liq_tau:
        if (i, j) not in [
            ("H2O", "H2O"),
            ("H2O", "C6H12"),
            ("C6H12", "H2O"),
            ("C6H12", "C6H12"),
        ]:
            assert str(model.state[1].Liq_tau[i, j].expr) == str(
                -log(model.state[1].Liq_G[i, j]) / model.state[1].Liq_alpha[i, j]
            )

    # Like species interactions
    assert ("Na+", "Na+") not in model.state[1].Liq_tau
    assert ("Na+", "H+") not in model.state[1].Liq_tau
    assert ("H+", "Na+") not in model.state[1].Liq_tau
    assert ("H+", "H+") not in model.state[1].Liq_tau
    assert ("Cl-", "Cl-") not in model.state[1].Liq_tau
    assert ("Cl-", "OH-") not in model.state[1].Liq_tau
    assert ("OH-", "Cl-") not in model.state[1].Liq_tau
    assert ("OH-", "OH-") not in model.state[1].Liq_tau

class TestStateBlockSymmetric(object):
    @pytest.fixture(scope="class")
    def model(self):
        m = ConcreteModel()
        m.params = GenericParameterBlock(**configuration)

        m.state = m.params.build_state_block([1])
        m.state[1].enth_mol_phase_comp = Var(m.params.phase_list, m.params.component_list)

        # Need to set a value of T for checking expressions later
        m.state[1].temperature.set_value(300)

        return m

    @pytest.mark.unit
    def test_common(self, model):
        # Reference state composition
        _test_common(model)
        _test_constant_alpha(model)
        _test_G_constant_alpha_two_parameter_tau(model)
        _test_two_parameter_tau(model)

        assert isinstance(model.state[1].Liq_vol_mol_solvent, Expression)
        assert len(model.state[1].Liq_vol_mol_solvent) == 1
        assert str(model.state[1].Liq_vol_mol_solvent.expr) == str(
            (
                1 / (42.0 * pyunits.mol / pyunits.m**3)
                * model.state[1].mole_frac_phase_comp_true["Liq", "H2O"]
                + 1 / (42.0 * pyunits.mol / pyunits.m**3)
                * model.state[1].mole_frac_phase_comp_true["Liq", "C6H12"]
            ) / ( 
                model.state[1].mole_frac_phase_comp_true["Liq", "H2O"]
                + model.state[1].mole_frac_phase_comp_true["Liq", "C6H12"]
            )
        )

        assert isinstance(model.state[1].Liq_relative_permittivity_solvent, Expression)
        assert len(model.state[1].Liq_relative_permittivity_solvent) == 1
        assert str(model.state[1].Liq_relative_permittivity_solvent.expr) == str(
            (
                model.state[1].mole_frac_phase_comp_true["Liq", "H2O"]
                * model.params.get_component("H2O").relative_permittivity_liq_comp
                * model.params.get_component("H2O").mw
                + model.state[1].mole_frac_phase_comp_true["Liq", "C6H12"]
                * model.params.get_component("C6H12").relative_permittivity_liq_comp
                * model.params.get_component("C6H12").mw
            ) / (
                model.state[1].mole_frac_phase_comp_true["Liq", "H2O"]
                * model.params.get_component("H2O").mw
                + model.state[1].mole_frac_phase_comp_true["Liq", "C6H12"]
                * model.params.get_component("C6H12").mw
            )
        )

        assert isinstance(model.state[1].Liq_ionic_strength_ref, Expression)
        assert len(model.state[1].Liq_ionic_strength_ref) == 1
        assert str(model.state[1].Liq_ionic_strength_ref.expr) == str(0.0)

        assert isinstance(model.state[1].Liq_log_gamma_lc_I0, Expression)
        assert len(model.state[1].Liq_log_gamma_lc_I0) == len(_all_components_set)
        for k in model.state[1].Liq_log_gamma_lc_I0:
            assert k in _all_components_set
        G = getattr(model.state[1],"Liq_G")
        tau = getattr(model.state[1],"Liq_tau")
        for k in _solvent_set:
            assert str(model.state[1].Liq_log_gamma_lc_I0[k].expr) == "0.0"
        for k in _charged_components_set:
            z = abs(model.params.get_component(k).config.charge)
            assert str(model.state[1].Liq_log_gamma_lc_I0[k].expr) == str(
                z * (G[k,"H2O"] * tau[k,"H2O"] + tau["H2O", k])
            )

        expr = model.state[1].Liq_log_gamma_born
        for k in expr:
            assert k in _all_components_set
            if k in _uncharged_components_set:
                assert str(expr[k].expr) == str(0.0)

            else:
                z = abs(model.params.get_component(k).config.charge)
                rb = model.params.get_component(k).born_radius
                tmp = str(
                    (z * Constants.elemental_charge)**2
                    / (
                        8
                        * Constants.pi
                        * Constants.vacuum_electric_permittivity
                        * Constants.boltzmann_constant
                        * model.state[1].temperature
                        * rb
                    ) * (
                        1 / model.state[1].Liq_relative_permittivity_solvent
                        - 1 / model.params.get_component("H2O").relative_permittivity_liq_comp
                    )
                )
                assert str(expr[k].expr) == tmp

        # The Poynting term is set to zero by default
        expr = model.state[1].Liq_log_gamma_poynting
        assert isinstance(expr, Expression)
        assert len(expr) == len(_all_components_set)
        for k in expr:
            assert k in _all_components_set
            assert str(expr[k].expr) == "0.0"

    @pytest.mark.unit
    def test_alpha(self, model):
        _test_constant_alpha(model)

    # @pytest.mark.unit
    # def test_G(self, model):
    #     _test_G_constant_alpha_and_tau(model)

    # @pytest.mark.unit
    # def test_tau(self, model):
    #     _test_constant_tau(model)