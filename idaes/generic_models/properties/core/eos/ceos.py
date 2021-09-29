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
Methods for cubic equations of state.

Currently only supports liquid and vapor phases
"""
import os
from enum import Enum

from pyomo.environ import (exp,
                           Expression,
                           ExternalFunction,
                           log,
                           Param,
                           Reals,
                           sqrt,
                           Var)
from pyomo.common.config import ConfigBlock, ConfigValue, In

from idaes.core.util.exceptions import PropertyNotSupportedError
from idaes.generic_models.properties.core.generic.utility import (
    get_method, get_component_object as cobj)
from idaes.core.util.math import safe_log
from .eos_base import EoSBase
from idaes import bin_directory
import idaes.logger as idaeslog
from idaes.core.util.exceptions import \
    BurntToast, ConfigurationError, PropertyNotSupportedError


# Set up logger
_log = idaeslog.getLogger(__name__)


# Set path to root finder .so file
_so = os.path.join(bin_directory, "cubic_roots.so")


def cubic_roots_available():
    """Make sure the compiled cubic root functions are available. Yes, in
    Windows the .so extention is still used.
    """
    return os.path.isfile(_so)


class CubicType(Enum):
    PR = 0
    SRK = 1


class MixingRuleA(Enum):
    default = 0


class MixingRuleB(Enum):
    default = 0


EoS_param = {
        CubicType.PR: {'u': 2, 'w': -1, 'omegaA': 0.45724, 'coeff_b': 0.07780},
        CubicType.SRK: {'u': 1, 'w': 0, 'omegaA': 0.42748, 'coeff_b': 0.08664}
        }


CubicConfig = ConfigBlock()
CubicConfig.declare("type", ConfigValue(
    domain=In(CubicType),
    description="Equation of state to use",
    doc="Enum indicating type of cubic equation of state to use."))


class Cubic(EoSBase):

    @staticmethod
    def common(b, pobj):
        # TODO: determine if Henry's Law applies to Cubic EoS systems
        # For now, raise an exception if found
        # Follow on questions:
        # If Henry's law is used for a component, how does that effect
        # calculating A, B and phi?
        for j in b.component_list:
            cobj = b.params.get_component(j)
            if (cobj.config.henry_component is not None and
                    pobj.local_name in cobj.config.henry_component):
                raise PropertyNotSupportedError(
                    "{} Cubic equations of state do not support Henry's "
                    "components [{}, {}].".format(b.name, pobj.local_name, j))

        ctype = pobj._cubic_type
        cname = pobj.config.equation_of_state_options["type"].name
        import pdb
        #pdb.set_trace()
        if hasattr(b, cname+"_fw"):
            # Common components already constructed by previous phase
            return
        
        # Create expressions for coefficients
        def func_fw(m, j):
            cobj = m.params.get_component(j)
            try:
                return pobj.config.equation_of_state_options["fw"](cobj.omega)
            except (KeyError, TypeError):
                if ctype == CubicType.PR:
                    return 0.37464 + 1.54226*cobj.omega - \
                           0.26992*cobj.omega**2
                elif ctype == CubicType.SRK:
                    return 0.48 + 1.574*cobj.omega - \
                           0.176*cobj.omega**2
                else:
                    raise BurntToast(
                            "{} received unrecognized cubic type. This should "
                            "never happen, so please contact the IDAES developers "
                            "with this bug.".format(b.name))

        b.add_component(cname+'_fw',
                        Expression(b.component_list,
                                   rule=func_fw,
                                   doc='EoS S factor'))
        
        def rule_alpha(m,j):
            alpha_func = get_alpha_func(pobj)
            cobj = m.params.get_component(j)
            fw = getattr(m, cname+"_fw")
            return alpha_func(m.temperature,fw[j] , cobj)
        
        b.add_component(cname+'_alpha',
                Expression(b.component_list,
                           rule=rule_alpha,
                           doc='Component alpha coefficient'))
                
        
        def rule_a(m, j):
            cobj = m.params.get_component(j)
            alphaj = getattr(m, cname+"_alpha")[j]
            return a_func(EoS_param[ctype]['omegaA'], Cubic.gas_constant(b), 
                          alphaj, cobj)
        
        b.add_component(cname+'_a',
                        Expression(b.component_list,
                                   rule=rule_a,
                                   doc='Component a coefficient'))

        def rule_b(m, j):
            cobj = m.params.get_component(j)
            return b_func(EoS_param[ctype]['coeff_b'],
                          Cubic.gas_constant(b), cobj)
        b.add_component(cname+'_b',
                        Expression(b.component_list,
                                   rule=rule_b,
                                   doc='Component b coefficient'))

        def rule_am(m, p):
            try:
                rule = m.params.get_phase(p).config.equation_of_state_options[
                    "mixing_rule_a"]
            except (KeyError, TypeError):
                rule = MixingRuleA.default

            a = getattr(m, cname+"_a")

            if rule == MixingRuleA.default:
                return rule_am_default(m, cname, a, p)
            else:
                raise ConfigurationError(
                    "{} Unrecognized option for Equation of State "
                    "mixing_rule_a: {}. Must be an instance of MixingRuleA "
                    "Enum.".format(m.name, rule))
        b.add_component(cname+'_am',
                        Expression(b.phase_list, rule=rule_am))

        def rule_bm(m, p):
            try:
                rule = m.params.get_phase(p).config.equation_of_state_options[
                    "mixing_rule_b"]
            except (KeyError, TypeError):
                rule = MixingRuleB.default

            b = getattr(m, cname+"_b")
            if rule == MixingRuleB.default:
                return rule_bm_default(m, b, p)
            else:
                raise ConfigurationError(
                    "{} Unrecognized option for Equation of State "
                    "mixing_rule_a: {}. Must be an instance of MixingRuleB "
                    "Enum.".format(m.name, rule))

        b.add_component(cname+'_bm',
                        Expression(b.phase_list, rule=rule_bm))

        def rule_A(m, p):
            am = getattr(m, cname+"_am")
            return (am[p]*m.pressure /
                    (Cubic.gas_constant(b)*m.temperature)**2)
        b.add_component(cname+'_A',
                        Expression(b.phase_list, rule=rule_A))

        def rule_B(m, p):
            bm = getattr(m, cname+"_bm")
            return (bm[p]*m.pressure /
                    (Cubic.gas_constant(b)*m.temperature))
        b.add_component(cname+'_B',
                        Expression(b.phase_list, rule=rule_B))

        def rule_delta(m, p, i):
            # See pg. 145 in Properties of Gases and Liquids
            a = getattr(m, cname+"_a")
            am = getattr(m, cname+"_am")
            kappa = getattr(m.params, cname+"_kappa")
            return (2*sqrt(a[i])/am[p] *
                    sum(m.mole_frac_phase_comp[p, j]*sqrt(a[j]) *
                        (1-kappa[i, j])
                        for j in b.components_in_phase(p)))
        b.add_component(cname+"_delta",
                        Expression(b.phase_component_set,
                                   rule=rule_delta))
               
        # TODO this also depends on the mixing rule
        def rule_dadT(m, p):
            # See pg. 102 in Properties of Gases and Liquids
            alpha = getattr(m, cname+"_alpha")
            a = getattr(m, cname+"_a")
            fw = getattr(m, cname+"_fw")
            kappa = getattr(m.params, cname+"_kappa")
            try:
                dalpha_dT = pobj.config.equation_of_state_options["dalpha_dT"]
            except (KeyError, TypeError):
                dalpha_dT = dalpha_dT_default
            # In case we support temperature dependence for kappa in the future
            def dkappaijdT(i,j):
                return 0
            
            def daijdT(i,j):
                return (sqrt(a[i]*a[j])
                        * (-dkappaijdT(i,j) + (1-kappa[i,j])/2
                           * (1/alpha[i]*dalpha_dT(m.temperature,fw[i],
                                    m.params.get_component(i))
                              + 1/alpha[j]*dalpha_dT(m.temperature,fw[j],
                                    m.params.get_component(j)))
                           )
                        )
            return sum(sum(m.mole_frac_phase_comp[p, i] *
                              m.mole_frac_phase_comp[p, j] * daijdT(i,j)
                          for j in m.components_in_phase(p))
                       for i in m.components_in_phase(p))
        
        b.add_component(cname+"_dadT",
                        Expression(b.phase_list,
                                   rule=rule_dadT))

        # Add components at equilibrium state if required
        if (b.params.config.phases_in_equilibrium is not None and
                (not b.config.defined_state or b.always_flash)):
            def rule_alpha_eq(m, p1, p2, j):
                alpha_func = get_alpha_func(pobj)
                cobj = m.params.get_component(j)
                fw = getattr(m, cname+"_fw")
                return alpha_func(m._teq[p1, p2],fw[j],cobj)
            
            b.add_component('_'+cname+'_alpha_eq',
                    Expression(b.params._pe_pairs,
                               b.component_list,
                               rule=rule_alpha_eq,
                               doc='Component alpha coefficient at Teq'))
            
            def rule_a_eq(m, p1, p2, j):
                cobj = m.params.get_component(j)
                alphaj_eq = getattr(m, "_"+cname+"_alpha_eq")[p1,p2,j]
                return a_func(EoS_param[ctype]['omegaA'], Cubic.gas_constant(b), 
                              alphaj_eq, cobj)
            
            b.add_component('_'+cname+'_a_eq',
                            Expression(b.params._pe_pairs,
                                       b.component_list,
                                       rule=rule_a_eq,
                                       doc='Component a coefficient at Teq'))

            def rule_am_eq(m, p1, p2, p3):
                try:
                    rule = m.params.get_phase(p3).config.equation_of_state_options[
                        "mixing_rule_a"]
                except (KeyError, TypeError):
                    rule = MixingRuleA.default

                a = getattr(m, "_"+cname+"_a_eq")
                if rule == MixingRuleA.default:
                    return rule_am_default(m, cname, a, p3, (p1, p2))
                else:
                    raise ConfigurationError(
                        "{} Unrecognized option for Equation of State "
                        "mixing_rule_a: {}. Must be an instance of MixingRuleA "
                        "Enum.".format(m.name, rule))
            b.add_component('_'+cname+'_am_eq',
                            Expression(b.params._pe_pairs,
                                       b.phase_list,
                                       rule=rule_am_eq))

            def rule_A_eq(m, p1, p2, p3):
                am_eq = getattr(m, "_"+cname+"_am_eq")
                return (am_eq[p1, p2, p3]*m.pressure /
                        (Cubic.gas_constant(b)*m._teq[p1, p2])**2)
            b.add_component('_'+cname+'_A_eq',
                            Expression(b.params._pe_pairs,
                                       b.phase_list,
                                       rule=rule_A_eq))

            def rule_B_eq(m, p1, p2, p3):
                bm = getattr(m, cname+"_bm")
                return (bm[p3]*m.pressure /
                        (Cubic.gas_constant(b)*m._teq[p1, p2]))
            b.add_component('_'+cname+'_B_eq',
                            Expression(b.params._pe_pairs,
                                       b.phase_list,
                                       rule=rule_B_eq))

            def rule_delta_eq(m, p1, p2, p3, i):
                # See pg. 145 in Properties of Gases and Liquids
                a = getattr(m, "_"+cname+"_a_eq")
                am = getattr(m, "_"+cname+"_am_eq")
                kappa = getattr(m.params, cname+"_kappa")
                return (2*sqrt(a[p1, p2, i])/am[p1, p2, p3] *
                        sum(m.mole_frac_phase_comp[p3, j]*sqrt(a[p1, p2, j]) *
                            (1-kappa[i, j])
                            for j in m.components_in_phase(p3)))
            b.add_component("_"+cname+"_delta_eq",
                            Expression(b.params._pe_pairs,
                                       b.phase_component_set,
                                       rule=rule_delta_eq))

        # Set up external function calls
        b.add_component("_"+cname+"_ext_func_param",
                        Param(default=ctype.value))
        b.add_component("_"+cname+"_proc_Z_liq",
                        ExternalFunction(library=_so,
                                         function="ceos_z_liq"))
        b.add_component("_"+cname+"_proc_Z_vap",
                        ExternalFunction(library=_so,
                                         function="ceos_z_vap"))

    @staticmethod
    def calculate_scaling_factors(b, pobj):
        pass

    @staticmethod
    def build_parameters(b):
        b._cubic_type = b.config.equation_of_state_options["type"]
        cname = b._cubic_type.name
        param_block = b.parent_block()

        if hasattr(param_block, cname+"_kappa"):
            # Common components already constructed by previous phase
            return

        kappa_data = param_block.config.parameter_data[cname+"_kappa"]
        param_block.add_component(
            cname+'_kappa',
            Var(param_block.component_list,
                param_block.component_list,
                within=Reals,
                initialize=kappa_data,
                doc=cname+' binary interaction parameters',
                units=None))

    @staticmethod
    def compress_fact_phase(b, p):
        pobj = b.params.get_phase(p)
        cname = pobj._cubic_type.name
        A = getattr(b, cname+"_A")
        B = getattr(b, cname+"_B")
        f = getattr(b, "_"+cname+"_ext_func_param")
        if pobj.is_vapor_phase():
            proc = getattr(b, "_"+cname+"_proc_Z_vap")
        elif pobj.is_liquid_phase():
            proc = getattr(b, "_"+cname+"_proc_Z_liq")
        else:
            raise PropertyNotSupportedError(_invalid_phase_msg(b.name, p))
        return proc(f, A[p], B[p])

    @staticmethod
    def dens_mass_phase(b, p):
        return b.dens_mol_phase[p]*b.mw_phase[p]

    @staticmethod
    def dens_mol_phase(b, p):
        pobj = b.params.get_phase(p)
        if pobj.is_vapor_phase() or pobj.is_liquid_phase():
            return b.pressure/(
                Cubic.gas_constant(b)*b.temperature*b.compress_fact_phase[p])
        else:
            raise PropertyNotSupportedError(_invalid_phase_msg(b.name, p))

    # TODO: Need to add functions to calculate cp and cv


    @staticmethod
    def energy_internal_mol_phase(blk, p):
        pobj = blk.params.get_phase(p)
        if not (pobj.is_vapor_phase() or pobj.is_liquid_phase()):
            raise PropertyNotSupportedError(_invalid_phase_msg(blk.name, p))

        cname = pobj._cubic_type.name
        am = getattr(blk, cname+"_am")[p]
        bm = getattr(blk, cname+"_bm")[p]
        B = getattr(blk, cname+"_B")[p]
        dadT = getattr(blk, cname+"_dadT")[p]
        Z = blk.compress_fact_phase[p]

        EoS_u = EoS_param[pobj._cubic_type]['u']
        EoS_w = EoS_param[pobj._cubic_type]['w']
        EoS_p = sqrt(EoS_u**2 - 4*EoS_w)

        # Derived from equation on pg. 120 in Properties of Gases and Liquids
        # Departure function for U is similar to H minus the RT(Z-1) term
        return (((blk.temperature*dadT - am) *
                 safe_log((2*Z + B*(EoS_u+EoS_p)) / (2*Z + B*(EoS_u-EoS_p)),
                          eps=1e-8)) / (bm*EoS_p) +
                sum(blk.mole_frac_phase_comp[p, j] *
                    EoSBase.energy_internal_mol_ig_comp_pure(blk, j)
                    for j in blk.components_in_phase(p)))

    @staticmethod
    def energy_internal_mol_phase_comp(blk, p, j):
        pobj = blk.params.get_phase(p)
        if not (pobj.is_vapor_phase() or pobj.is_liquid_phase()):
            raise PropertyNotSupportedError(_invalid_phase_msg(blk.name, p))

        cname = pobj._cubic_type.name
        am = getattr(blk, cname+"_am")[p]
        bm = getattr(blk, cname+"_bm")[p]
        B = getattr(blk, cname+"_B")[p]
        dadT = getattr(blk, cname+"_dadT")[p]
        Z = blk.compress_fact_phase[p]

        EoS_u = EoS_param[pobj._cubic_type]['u']
        EoS_w = EoS_param[pobj._cubic_type]['w']
        EoS_p = sqrt(EoS_u**2 - 4*EoS_w)

        # Derived from equation on pg. 120 in Properties of Gases and Liquids
        # Departure function for U is similar to H minus the RT(Z-1) term
        return (((blk.temperature*dadT - am) *
                 safe_log((2*Z + B*(EoS_u+EoS_p)) / (2*Z + B*(EoS_u-EoS_p)),
                          eps=1e-8)) / (bm*EoS_p) +
                EoSBase.energy_internal_mol_ig_comp_pure(blk, j))

    @staticmethod
    def enth_mol_phase(blk, p):
        pobj = blk.params.get_phase(p)
        if not (pobj.is_vapor_phase() or pobj.is_liquid_phase()):
            raise PropertyNotSupportedError(_invalid_phase_msg(blk.name, p))

        cname = pobj._cubic_type.name
        am = getattr(blk, cname+"_am")[p]
        bm = getattr(blk, cname+"_bm")[p]
        B = getattr(blk, cname+"_B")[p]
        dadT = getattr(blk, cname+"_dadT")[p]
        Z = blk.compress_fact_phase[p]

        EoS_u = EoS_param[pobj._cubic_type]['u']
        EoS_w = EoS_param[pobj._cubic_type]['w']
        EoS_p = sqrt(EoS_u**2 - 4*EoS_w)

        # Derived from equation on pg. 120 in Properties of Gases and Liquids
        return (((blk.temperature*dadT - am) *
                 safe_log((2*Z + B*(EoS_u+EoS_p)) / (2*Z + B*(EoS_u-EoS_p)),
                          eps=1e-8) +
                 Cubic.gas_constant(blk)*blk.temperature*(Z-1)*bm*EoS_p) /
                (bm*EoS_p) + sum(blk.mole_frac_phase_comp[p, j] *
                                 get_method(blk, "enth_mol_ig_comp", j)(
                                            blk, cobj(blk, j), blk.temperature)
                                 for j in blk.components_in_phase(p)))

    @staticmethod
    def enth_mol_phase_comp(blk, p, j):
        pobj = blk.params.get_phase(p)
        if not (pobj.is_vapor_phase() or pobj.is_liquid_phase()):
            raise PropertyNotSupportedError(_invalid_phase_msg(blk.name, p))

        cname = pobj._cubic_type.name
        am = getattr(blk, cname+"_am")[p]
        bm = getattr(blk, cname+"_bm")[p]
        B = getattr(blk, cname+"_B")[p]
        dadT = getattr(blk, cname+"_dadT")[p]
        Z = blk.compress_fact_phase[p]

        EoS_u = EoS_param[pobj._cubic_type]['u']
        EoS_w = EoS_param[pobj._cubic_type]['w']
        EoS_p = sqrt(EoS_u**2 - 4*EoS_w)

        # Derived from equation on pg. 120 in Properties of Gases and Liquids
        return (((blk.temperature*dadT - am) *
                 safe_log((2*Z + B*(EoS_u+EoS_p)) / (2*Z + B*(EoS_u-EoS_p)),
                          eps=1e-8) +
                 Cubic.gas_constant(blk)*blk.temperature*(Z-1)*bm*EoS_p) /
                (bm*EoS_p) + get_method(blk, "enth_mol_ig_comp", j)(
                                        blk, cobj(blk, j), blk.temperature))

    @staticmethod
    def entr_mol_phase(blk, p):
        pobj = blk.params.get_phase(p)
        if not (pobj.is_vapor_phase() or pobj.is_liquid_phase()):
            raise PropertyNotSupportedError(_invalid_phase_msg(blk.name, p))

        cname = pobj._cubic_type.name
        bm = getattr(blk, cname+"_bm")[p]
        B = getattr(blk, cname+"_B")[p]
        dadT = getattr(blk, cname+"_dadT")[p]
        Z = blk.compress_fact_phase[p]

        EoS_u = EoS_param[pobj._cubic_type]['u']
        EoS_w = EoS_param[pobj._cubic_type]['w']
        EoS_p = sqrt(EoS_u**2 - 4*EoS_w)

        # See pg. 102 in Properties of Gases and Liquids
        return ((Cubic.gas_constant(blk)*safe_log((Z-B)/Z, eps=1e-8)*bm*EoS_p +
                 Cubic.gas_constant(blk) *
                 safe_log(Z*blk.params.pressure_ref/blk.pressure, eps=1e-8) *
                 bm*EoS_p +
                 dadT*safe_log((2*Z + B*(EoS_u + EoS_p)) /
                               (2*Z + B*(EoS_u - EoS_p)),
                               eps=1e-8)) /
                (bm*EoS_p) + sum(blk.mole_frac_phase_comp[p, j] *
                                 get_method(blk, "entr_mol_ig_comp", j)(
                                     blk, cobj(blk, j), blk.temperature)
                                 for j in blk.components_in_phase(p)))

    @staticmethod
    def entr_mol_phase_comp(blk, p, j):
        pobj = blk.params.get_phase(p)
        if not (pobj.is_vapor_phase() or pobj.is_liquid_phase()):
            raise PropertyNotSupportedError(_invalid_phase_msg(blk.name, p))

        cname = pobj._cubic_type.name
        bm = getattr(blk, cname+"_bm")[p]
        B = getattr(blk, cname+"_B")[p]
        dadT = getattr(blk, cname+"_dadT")[p]
        Z = blk.compress_fact_phase[p]

        EoS_u = EoS_param[pobj._cubic_type]['u']
        EoS_w = EoS_param[pobj._cubic_type]['w']
        EoS_p = sqrt(EoS_u**2 - 4*EoS_w)

        # See pg. 102 in Properties of Gases and Liquids
        return (((Cubic.gas_constant(blk)*safe_log((Z-B)/Z, eps=1e-8) *
                  bm*EoS_p +
                  Cubic.gas_constant(blk) *
                  safe_log(Z*blk.params.pressure_ref /
                           (blk.mole_frac_phase_comp[p, j]*blk.pressure),
                           eps=1e-8) *
                  bm*EoS_p +
                  dadT*safe_log((2*Z + B*(EoS_u + EoS_p)) /
                                (2*Z + B*(EoS_u - EoS_p)),
                                eps=1e-8)) /
                (bm*EoS_p)) + get_method(blk, "entr_mol_ig_comp", j)(
                                      blk, cobj(blk, j), blk.temperature))

    @staticmethod
    def fug_phase_comp(b, p, j):
        pobj = b.params.get_phase(p)
        if pobj.is_vapor_phase() or pobj.is_liquid_phase():
            return (b.mole_frac_phase_comp[p, j] *
                    b.pressure *
                    b.fug_coeff_phase_comp[p, j])
        else:
            raise PropertyNotSupportedError(_invalid_phase_msg(b.name, p))

    @staticmethod
    def fug_phase_comp_eq(b, p, j, pp):
        pobj = b.params.get_phase(p)
        if pobj.is_vapor_phase() or pobj.is_liquid_phase():
            return (b.mole_frac_phase_comp[p, j] *
                    b.pressure *
                    exp(_log_fug_coeff_phase_comp_eq(b, p, j, pp)))
        else:
            raise PropertyNotSupportedError(_invalid_phase_msg(b.name, p))

    @staticmethod
    def log_fug_phase_comp_eq(b, p, j, pp):
        pobj = b.params.get_phase(p)
        if pobj.is_vapor_phase() or pobj.is_liquid_phase():
            return (log(b.mole_frac_phase_comp[p, j]) +
                    log(b.pressure/b.params.pressure_ref) +
                    _log_fug_coeff_phase_comp_eq(b, p, j, pp))
        else:
            raise PropertyNotSupportedError(_invalid_phase_msg(b.name, p))

    @staticmethod
    def fug_coeff_phase_comp(blk, p, j):
        pobj = blk.params.get_phase(p)
        ctype = pobj._cubic_type

        if not (pobj.is_vapor_phase() or pobj.is_liquid_phase()):
            raise PropertyNotSupportedError(_invalid_phase_msg(blk.name, p))

        cname = pobj._cubic_type.name
        b = getattr(blk, cname+"_b")[j]
        bm = getattr(blk, cname+"_bm")[p]
        A = getattr(blk, cname+"_A")[p]
        B = getattr(blk, cname+"_B")[p]
        delta = getattr(blk, cname+"_delta")[p, j]
        Z = blk.compress_fact_phase[p]

        return exp(_log_fug_coeff_method(A, b, bm, B, delta, Z, ctype))

    @staticmethod
    def fug_coeff_phase_comp_eq(blk, p, j, pp):
        return exp(_log_fug_coeff_phase_comp_eq(blk, p, j, pp))

    #TODO all these bubble and dew point calculations hardcode mixing rules
    #which is a landmine for whoever tries to implement the first option
    @staticmethod
    def log_fug_phase_comp_Tbub(blk, p, j, pp):
        pobj = blk.params.get_phase(p)
        ctype = pobj._cubic_type
        cname = pobj.config.equation_of_state_options["type"].name

        if pobj.is_liquid_phase():
            x = blk.mole_frac_comp
            xidx = ()
        elif pobj.is_vapor_phase():
            x = blk._mole_frac_tbub
            xidx = pp
        else:
            raise BurntToast("{} non-vapor or liquid phase called for bubble "
                             "temperature calculation. This should never "
                             "happen, so please contact the IDAES developers "
                             "with this bug.".format(blk.name))

        def a(k):
            cobj = blk.params.get_component(k)
            fw = getattr(m, cname+"_fw")
            alpha_func = get_alpha_func(pobj)
            alphak = alpha_func(blk.temperature_bubble[pp],fw[k],cobj)
            return a_func(EoS_param[ctype]['omegaA'], Cubic.gas_constant(blk), 
                          alphak, cobj)

        kappa = getattr(blk.params, cname+"_kappa")
        am = sum(sum(x[xidx, i]*x[xidx, j]*sqrt(a(i)*a(j))*(1-kappa[i, j])
                     for j in blk.component_list)
                 for i in blk.component_list)

        b = getattr(blk, cname+"_b")
        bm = sum(x[xidx, i]*b[i] for i in blk.component_list)

        A = am*blk.pressure/(Cubic.gas_constant(blk) *
                             blk.temperature_bubble[pp])**2
        B = bm*blk.pressure/(Cubic.gas_constant(blk) *
                             blk.temperature_bubble[pp])

        delta = (2*sqrt(a(j))/am * sum(x[xidx, i]*sqrt(a(i))*(1-kappa[j, i])
                                       for i in blk.component_list))

        f = getattr(blk, "_"+cname+"_ext_func_param")
        if pobj.is_vapor_phase():
            proc = getattr(blk, "_"+cname+"_proc_Z_vap")
        elif pobj.is_liquid_phase():
            proc = getattr(blk, "_"+cname+"_proc_Z_liq")

        Z = proc(f, A, B)

        if pobj.is_vapor_phase():
            mole_frac = blk._mole_frac_tbub[pp[0], pp[1], j]
        else:
            mole_frac = blk.mole_frac_comp[j]

        return (_log_fug_coeff_method(A, b[j], bm, B, delta, Z, ctype) +
                log(mole_frac) + log(blk.pressure/blk.pressure._units))

    @staticmethod
    def log_fug_phase_comp_Tdew(blk, p, j, pp):
        pobj = blk.params.get_phase(p)
        ctype = pobj._cubic_type
        cname = pobj.config.equation_of_state_options["type"].name

        if pobj.is_liquid_phase():
            x = blk._mole_frac_tdew
            xidx = pp
        elif pobj.is_vapor_phase():
            x = blk.mole_frac_comp
            xidx = ()
        else:
            raise BurntToast("{} non-vapor or liquid phase called for bubble "
                             "temperature calculation. This should never "
                             "happen, so please contact the IDAES developers "
                             "with this bug.".format(blk.name))

        def a(k):
            cobj = blk.params.get_component(k)
            fw = getattr(m, cname+"_fw")
            alpha_func = get_alpha_func(pobj)
            alphak = alpha_func(blk.temperature_dw[pp],fw[k],cobj)
            return a_func(EoS_param[ctype]['omegaA'], Cubic.gas_constant(blk), 
                          alphak, cobj)

        kappa = getattr(blk.params, cname+"_kappa")
        am = sum(sum(x[xidx, i]*x[xidx, j]*sqrt(a(i)*a(j))*(1-kappa[i, j])
                     for j in blk.component_list)
                 for i in blk.component_list)

        b = getattr(blk, cname+"_b")
        bm = sum(x[xidx, i]*b[i] for i in blk.component_list)

        A = am*blk.pressure/(Cubic.gas_constant(blk) *
                             blk.temperature_dew[pp])**2
        B = bm*blk.pressure/(Cubic.gas_constant(blk) *
                             blk.temperature_dew[pp])

        delta = (2*sqrt(a(j))/am * sum(x[xidx, i]*sqrt(a(i))*(1-kappa[j, i])
                                       for i in blk.component_list))

        f = getattr(blk, "_"+cname+"_ext_func_param")
        if pobj.is_vapor_phase():
            proc = getattr(blk, "_"+cname+"_proc_Z_vap")
        elif pobj.is_liquid_phase():
            proc = getattr(blk, "_"+cname+"_proc_Z_liq")

        Z = proc(f, A, B)

        if pobj.is_vapor_phase():
            mole_frac = blk.mole_frac_comp[j]
        else:
            mole_frac = blk._mole_frac_tdew[pp[0], pp[1], j]

        return (_log_fug_coeff_method(A, b[j], bm, B, delta, Z, ctype) +
                log(mole_frac) + log(blk.pressure/blk.pressure._units))

    @staticmethod
    def log_fug_phase_comp_Pbub(blk, p, j, pp):
        pobj = blk.params.get_phase(p)
        ctype = pobj._cubic_type
        cname = pobj.config.equation_of_state_options["type"].name

        if pobj.is_liquid_phase():
            x = blk.mole_frac_comp
            xidx = ()
        elif pobj.is_vapor_phase():
            x = blk._mole_frac_pbub
            xidx = pp
        else:
            raise BurntToast("{} non-vapor or liquid phase called for bubble "
                             "temperature calculation. This should never "
                             "happen, so please contact the IDAES developers "
                             "with this bug.".format(blk.name))

        a = getattr(blk, cname+"_a")
        kappa = getattr(blk.params, cname+"_kappa")
        am = sum(sum(x[xidx, i]*x[xidx, j] *
                     sqrt(a[i]*a[j])*(1-kappa[i, j])
                     for j in blk.component_list)
                 for i in blk.component_list)

        b = getattr(blk, cname+"_b")
        bm = sum(x[xidx, i]*b[i] for i in blk.component_list)

        A = am*blk.pressure_bubble[pp]/(Cubic.gas_constant(blk) *
                                        blk.temperature)**2
        B = bm*blk.pressure_bubble[pp]/(Cubic.gas_constant(blk) *
                                        blk.temperature)

        delta = (2*sqrt(a[j])/am * sum(x[xidx, i]*sqrt(a[i])*(1-kappa[j, i])
                                       for i in blk.component_list))

        f = getattr(blk, "_"+cname+"_ext_func_param")
        if pobj.is_vapor_phase():
            proc = getattr(blk, "_"+cname+"_proc_Z_vap")
        elif pobj.is_liquid_phase():
            proc = getattr(blk, "_"+cname+"_proc_Z_liq")

        Z = proc(f, A, B)

        if pobj.is_vapor_phase():
            mole_frac = blk._mole_frac_pbub[pp[0], pp[1], j]
        else:
            mole_frac = blk.mole_frac_comp[j]

        return (_log_fug_coeff_method(A, b[j], bm, B, delta, Z, ctype) +
                log(mole_frac) + log(blk.pressure_bubble[pp] /
                                     blk.pressure_bubble._units))

    @staticmethod
    def log_fug_phase_comp_Pdew(blk, p, j, pp):
        pobj = blk.params.get_phase(p)
        ctype = pobj._cubic_type
        cname = pobj.config.equation_of_state_options["type"].name

        if pobj.is_liquid_phase():
            x = blk._mole_frac_pdew
            xidx = pp
        elif pobj.is_vapor_phase():
            x = blk.mole_frac_comp
            xidx = ()
        else:
            raise BurntToast("{} non-vapor or liquid phase called for bubble "
                             "temperature calculation. This should never "
                             "happen, so please contact the IDAES developers "
                             "with this bug.".format(blk.name))

        a = getattr(blk, cname+"_a")
        kappa = getattr(blk.params, cname+"_kappa")
        am = sum(sum(x[xidx, i]*x[xidx, j] *
                     sqrt(a[i]*a[j])*(1-kappa[i, j])
                     for j in blk.component_list)
                 for i in blk.component_list)

        b = getattr(blk, cname+"_b")
        bm = sum(x[xidx, i]*b[i] for i in blk.component_list)

        A = am*blk.pressure_dew[pp]/(Cubic.gas_constant(blk) *
                                     blk.temperature)**2
        B = bm*blk.pressure_dew[pp]/(Cubic.gas_constant(blk)*blk.temperature)

        delta = (2*sqrt(a[j])/am * sum(x[xidx, i]*sqrt(a[i])*(1-kappa[j, i])
                                       for i in blk.component_list))

        f = getattr(blk, "_"+cname+"_ext_func_param")
        if pobj.is_vapor_phase():
            proc = getattr(blk, "_"+cname+"_proc_Z_vap")
        elif pobj.is_liquid_phase():
            proc = getattr(blk, "_"+cname+"_proc_Z_liq")

        Z = proc(f, A, B)

        if pobj.is_vapor_phase():
            mole_frac = blk.mole_frac_comp[j]
        else:
            mole_frac = blk._mole_frac_pdew[pp[0], pp[1], j]

        return (_log_fug_coeff_method(A, b[j], bm, B, delta, Z, ctype) +
                log(mole_frac) + log(blk.pressure_dew[pp] /
                                     blk.pressure_dew._units))

    @staticmethod
    def gibbs_mol_phase(b, p):
        return (b.enth_mol_phase[p] - b.entr_mol_phase[p]*b.temperature)

    @staticmethod
    def gibbs_mol_phase_comp(b, p, j):
        return (b.enth_mol_phase_comp[p, j] -
                b.entr_mol_phase_comp[p, j] *
                b.temperature)

    @staticmethod
    def vol_mol_phase(b, p):
        pobj = b.params.get_phase(p)
        if pobj.is_vapor_phase() or pobj.is_liquid_phase():
            return (Cubic.gas_constant(b)*b.temperature *
                    b.compress_fact_phase[p] /
                    b.pressure)
        else:
            raise PropertyNotSupportedError(_invalid_phase_msg(b.name, p))


def _invalid_phase_msg(name, phase):
    return ("{} received unrecognized phase name {}. Ideal property "
            "libray only supports Vap and Liq phases."
            .format(name, phase))


def _log_fug_coeff_phase_comp_eq(blk, p, j, pp):
    pobj = blk.params.get_phase(p)
    if not (pobj.is_vapor_phase() or pobj.is_liquid_phase()):
        raise PropertyNotSupportedError(_invalid_phase_msg(blk.name, p))

    cname = pobj._cubic_type.name
    b = getattr(blk, cname+"_b")
    bm = getattr(blk, cname+"_bm")
    Aeq = getattr(blk, "_"+cname+"_A_eq")
    Beq = getattr(blk, "_"+cname+"_B_eq")
    delta_eq = getattr(blk, "_"+cname+"_delta_eq")

    f = getattr(blk, "_"+cname+"_ext_func_param")
    if pobj.is_vapor_phase():
        proc = getattr(blk, "_"+cname+"_proc_Z_vap")
    elif pobj.is_liquid_phase():
        proc = getattr(blk, "_"+cname+"_proc_Z_liq")

    def Zeq(p):
        return proc(f, Aeq[pp, p], Beq[pp, p])

    return _log_fug_coeff_method(Aeq[pp, p], b[j], bm[p], Beq[pp, p],
                                 delta_eq[pp, p, j], Zeq(p), pobj._cubic_type)


def _log_fug_coeff_method(A, b, bm, B, delta, Z, cubic_type):
    u = EoS_param[cubic_type]['u']
    w = EoS_param[cubic_type]['w']
    p = sqrt(u**2 - 4*w)

    return ((b/bm*(Z-1)*(B*p) - safe_log(Z-B, eps=1e-8)*(B*p) +
             A*(b/bm - delta)*safe_log((2*Z + B*(u + p))/(2*Z + B*(u - p)),
                                       eps=1e-8)) /
            (B*p))


# -----------------------------------------------------------------------------
def alpha_func_default(T, fw, cobj):
    Tr = T /cobj.temperature_crit
    return (1+fw*(1-sqrt(Tr)))**2

def dalpha_dT_default(T,fw,cobj):
    Tr = T /cobj.temperature_crit
    return -fw/sqrt(T*cobj.temperature_crit)*(1+fw*(1-sqrt(Tr)))

def a_func(omegaA, R, alphaj, cobj):
    return (omegaA*((R * cobj.temperature_crit)**2/cobj.pressure_crit)
                    * alphaj)

def b_func(coeff_b, R, cobj):
            return coeff_b * R * cobj.temperature_crit/cobj.pressure_crit

# Mixing rules
def rule_am_default(m, cname, a, p, pp=()):
    k = getattr(m.params, cname+"_kappa")
    return sum(sum(
        m.mole_frac_phase_comp[p, i]*m.mole_frac_phase_comp[p, j] *
        sqrt(a[pp, i]*a[pp, j])*(1-k[i, j])
        for j in m.components_in_phase(p))
        for i in m.components_in_phase(p))


def rule_bm_default(m, b, p):
    return sum(m.mole_frac_phase_comp[p, i]*b[i]
               for i in m.components_in_phase(p))

def get_alpha_func(pobj):
    ctype = pobj._cubic_type
    try:
        alpha_func = pobj.config.equation_of_state_options["alpha"]
    except (KeyError, TypeError):
        alpha_func = alpha_func_default
    return alpha_func