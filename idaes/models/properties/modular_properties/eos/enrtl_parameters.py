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
Sub-methods for eNRTL activity coefficient method.

Includes temperature dependence rules for alpha and tau
"""
# TODO: Missing docstrings
# pylint: disable=missing-function-docstring

from pyomo.environ import Reals, units as pyunits, Var

from idaes.core.util.exceptions import BurntToast, ConfigurationError
import idaes.logger as idaeslog

# Set up logger
_log = idaeslog.getLogger(__name__)


class ConstantAlpha(object):
    """Class for methods assuming constant alpha"""

    @staticmethod
    def build_parameters(b):
        """Looks for user assigned value for alpha, and if not present uses
        a value of 0.3 for molecule-molecule interactions and 0.2 for molecule-
        ion and ion-ion interactions."""
        param_block = b.parent_block()

        # Get user provided values for alpha (if present)
        # TODO why is this looking on the parent block and not the phase block?
        try:
            alpha_data = param_block.config.parameter_data[b.local_name + "_alpha"]
        except KeyError:
            alpha_data = {}

        # Check for unused parameters in alpha_data
        for i, j in alpha_data.keys():
            if (i, j) not in b.component_pair_set_symmetric and (
                j,
                i,
            ) not in b.component_pair_set_symmetric:
                raise ConfigurationError(
                    "{} eNRTL alpha parameter provided for invalid "
                    "component pair {}. Please check typing and only provide "
                    "parameters for valid species pairs.".format(b.name, (i, j))
                )

        def alpha_init(b, i, j):
            if (i, j) in alpha_data.keys():
                v = alpha_data[(i, j)]
                # Check for non-symmetric value assignment
                if (j, i) in alpha_data.keys():
                    if alpha_data[(j, i)] != v:
                        raise ConfigurationError(
                            "{} eNRTL alpha parameter assigned non-symmetric "
                            "value for pair {}. Please assign only one value "
                            "for component pair.".format(b.name, (i, j))
                        )
                    else:
                        _log.info(
                            "eNRTL alpha value provided for both {} and "
                            "{}. It is only necessary to provide a "
                            "value for one of these due to symmetry.".format(
                                (i, j), (j, i)
                            )
                        )
            elif (j, i) in alpha_data.keys():
                v = alpha_data[(j, i)]
            elif (i in param_block.solvent_set or i in param_block.solute_set) and (
                j in param_block.solvent_set or j in param_block.solute_set
            ):
                # Molecular-molecular interaction, default value is 0.3
                v = 0.3
            else:
                # All other interactions have default value 0.2
                v = 0.2
            return v

        b.add_component(
            "alpha",
            Var(
                b.component_pair_set_symmetric,
                within=Reals,
                initialize=alpha_init,
                doc="Symmetric non-randomness parameters",
                units=pyunits.dimensionless,
            ),
        )

    @staticmethod
    def return_alpha_expression(b, pobj, i, j, T):
        """For the component pair (i, j), return the assigned (constant)
        value for alpha. If i==j, return the default value of 0.2.
        """
        if (i, j) in pobj.alpha:
            return pobj.alpha[i, j]
        elif (j, i) in pobj.alpha:
            return pobj.alpha[j, i]
        elif i == j:
            return 0.2
        else:
            raise BurntToast(
                "{} alpha rule encountered unexpected index {}. Please contact"
                "the IDAES Developers with this bug.".format(b.name, (i, j))
            )
    @staticmethod
    def return_dalpha_dT_expression(b, pobj, i, j, T):
        """Returns the derivative of alpha with respect to temperature.
        Since alpha is constant, the value returned is zero."""
        units = b.params.get_metadata().derived_units
        return 0 / units.TEMPERATURE


class ConstantTau(object):
    """Class for methods assuming constant tau"""

    @staticmethod
    def build_parameters(b):
        """Looks for a user-assigned value for tau, and, if absent, assigns
        the default value of 0."""
        param_block = b.parent_block()

        # Get user provided values for tau (if present)
        try:
            tau_data = param_block.config.parameter_data[b.local_name + "_tau"]
        except KeyError:
            tau_data = {}

        # Check for unused parameters in tau_data
        for i, j in tau_data.keys():
            if (i, j) not in b.component_pair_set:
                raise ConfigurationError(
                    "{} eNRTL tau parameter provided for invalid "
                    "component pair {}. Please check typing and only provide "
                    "parameters for valid species pairs.".format(b.name, (i, j))
                )

        def tau_init(b, i, j):
            if (i, j) in tau_data.keys():
                v = tau_data[(i, j)]
            else:
                # Default interaction value is 0
                v = 0
            return v

        b.add_component(
            "tau",
            Var(
                b.component_pair_set,
                within=Reals,
                initialize=tau_init,
                doc="Binary interaction energy parameters",
                units=pyunits.dimensionless,
            ),
        )

    @staticmethod
    def return_tau_expression(b, pobj, i, j, T):
        if (i, j) in pobj.tau:
            return pobj.tau[i, j]
        elif (j, i) in pobj.tau:
            return pobj.tau[j, i]
        elif i == j:
            return 0
        else:
            raise BurntToast(
                "{} tau rule encountered unexpected index {}. Please contact"
                "the IDAES Developers with this bug.".format(b.name, (i, j))
            )
        
    @staticmethod
    def return_dtau_dT_expression(b, pobj, i, j, T):
        if (i, j) in pobj.tau:
            return 0
        elif (j, i) in pobj.tau:
            return 0
        elif i == j:
            return 0
        else:
            raise BurntToast(
                "{} tau rule encountered unexpected index {}. Please contact"
                "the IDAES Developers with this bug.".format(b.name, (i, j))
            )

class TwoParameterTau(object):
    """Class in which Tau is broken into enthalpic and entropic components """

    @staticmethod
    def build_parameters(b):
        param_block = b.parent_block()
        units = param_block.get_metadata().derived_units

        # Get user provided values for tau (if present)
        try:
            tau_A_data = param_block.config.parameter_data[b.local_name + "_tau_A"]
        except KeyError:
            tau_A_data = {}

        try:
            tau_B_data = param_block.config.parameter_data[b.local_name + "_tau_B"]
        except KeyError:
            tau_B_data = {}

        # Check for unused parameters in tau_data
        for data in tau_A_data, tau_B_data:
            for i, j in data.keys():
                if (i, j) not in b.component_pair_set:
                    raise ConfigurationError(
                        "{} eNRTL tau parameter provided for invalid "
                        "component pair {}. Please check typing and only provide "
                        "parameters for valid species pairs.".format(b.name, (i, j))
                    )

        def tau_A_init(b, i, j):
            try:
                return tau_A_data[(i, j)]
            except KeyError:
                # Default interaction value is 0
                return 0
        
        def tau_B_init(b, i, j):
            try:
                return tau_B_data[(i, j)]
            except KeyError:
                # Default interaction value is 0
                return 0

        b.add_component(
            "tau_A",
            Var(
                b.component_pair_set,
                within=Reals,
                initialize=tau_A_init,
                doc="Binary interaction energy parameters",
                units=pyunits.dimensionless,
            ),
        )

        b.add_component(
            "tau_B",
            Var(
                b.component_pair_set,
                within=Reals,
                initialize=tau_B_init,
                doc="Binary interaction energy parameters",
                units=units["temperature"],
            ),
        )

    @staticmethod
    def return_tau_expression(b, pobj, i, j, T):
        if (i, j) in pobj.tau_A:
            return pobj.tau_A[i, j] + pobj.tau_B[i, j] / T
        elif i == j:
            return 0
        else:
            raise BurntToast(
                "{} tau rule encountered unexpected index {}. Please contact"
                "the IDAES Developers with this bug.".format(b.name, (i, j))
            )
        
    @staticmethod
    def return_dtau_dT_expression(b, pobj, i, j, T):
        if (i, j) in pobj.tau_A:
            return -pobj.tau_B[i, j] / T ** 2
        elif i == j:
            units = b.params.get_metadata().derived_units
            return 0 / units.TEMPERATURE
        else:
            raise BurntToast(
                "{} tau rule encountered unexpected index {}. Please contact"
                "the IDAES Developers with this bug.".format(b.name, (i, j))
            )