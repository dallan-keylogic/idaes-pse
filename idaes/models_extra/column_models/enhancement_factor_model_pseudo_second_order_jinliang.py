# Import Python libraries
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Pyomo libraries
import pyomo.opt
from pyomo.environ import (
    Block,
    ConcreteModel,
    value,
    Var,
    Reals,
    NonNegativeReals,
    Param,
    TransformationFactory,
    Constraint,
    Expression,
    Objective,
    SolverStatus,
    TerminationCondition,
    check_optimal_termination,
    assert_optimal_termination,
    exp,
    log,
    sqrt,
    units as pyunits,
    Set,
    Reference,
)
from pyomo.common.collections import ComponentSet, ComponentMap

from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.common.config import ConfigValue, Bool

# Import IDAES Libraries
from idaes.core.util.constants import Constants as CONST
from idaes.models_extra.column_models.solvent_column import PackedColumnData

from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import _fix_vars, _restore_fixedness
from idaes.core import declare_process_block_class, FlowsheetBlock, StateBlock
from idaes.core.util.exceptions import InitializationError
from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog

from idaes.core.solvers import use_idaes_solver_configuration_defaults
import idaes.core.util.scaling as iscale
from pyomo.util.subsystems import (
    create_subsystem_block,
)
from idaes.core.solvers.petsc import (
    _sub_problem_scaling_suffix,
)


def make_enhancement_factor_model(blk, lunits, kinetics="Putta"):
    """
    Enhancement factor based liquid phase mass transfer model.
    """
    assert kinetics in {"Luo", "Putta"}

    @blk.Expression(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        doc="Second order rate constant [m3/(mol.s)]",
    )
    def rate_constant(b, t, x):
        if x == b.liquid_phase.length_domain.last():
            return Expression.Skip
        else:
            T = pyunits.convert(
                b.liquid_phase.properties[t, x].temperature, to_units=pyunits.K
            )
            C_MEA = pyunits.convert(
                b.liquid_phase.properties[t, x].conc_mol_phase_comp_true["Liq", "MEA"],
                to_units=pyunits.mol / pyunits.m**3,
            )
            C_H2O = pyunits.convert(
                b.liquid_phase.properties[t, x].conc_mol_phase_comp_true["Liq", "H2O"],
                to_units=pyunits.mol / pyunits.m**3,
            )
            # Putta, Svendsen, Knuutila 2017 Eqn. 42
            if kinetics == "Putta":
                return (
                    pyunits.convert(
                        (
                            3.1732e9 * exp(-4936.6 * pyunits.K / T) * C_MEA * 1e-6
                            + 1.0882e8 * exp(-3900 * pyunits.K / T) * C_H2O * 1e-6
                        )
                        * ((pyunits.m) ** 6 / (pyunits.mol**2 * pyunits.s)),
                        to_units=1 / (lunits("time") * lunits("density_mole")),
                    )
                )
            elif kinetics == "Luo":
                return (
                    pyunits.convert(
                        (
                            2.003e10 * exp(-4742 * pyunits.K / T) * C_MEA * 1e-6
                            + 4.147e6 * exp(-3110 * pyunits.K / T) * C_H2O * 1e-6
                        )
                        * ((pyunits.m) ** 6 / (pyunits.mol**2 * pyunits.s)),
                        to_units=1 / (lunits("time") * lunits("density_mole")),
                    )
                )
            else:
                return AssertionError

    @blk.Expression(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        doc="Hatta number",
    )
    def hatta_number(b, t, x):
        if x == b.liquid_phase.length_domain.last():
            return Expression.Skip
        else:
            return (
                sqrt(
                    b.rate_constant[t, x]
                    * b.liquid_phase.properties[t, x].conc_mol_phase_comp_true[
                        "Liq", "MEA"
                    ]
                    * b.liquid_phase.properties[t, x].diffus_phase_comp["Liq", "CO2"]
                )
                / b.mass_transfer_coeff_liq[t, x, "CO2"]
            )

    blk.red_conc_CO2_bulk = Var(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        initialize=1,
        units=pyunits.dimensionless,
        bounds=(0, None),
        doc="""Reduced concentration of CO2,
                Driving force term in which
                Absorption implies conc_CO2_bulk < 1 and 
                Desorption implies conc_CO2_bulk > 1 """,
    )

    blk.red_conc_CO2_equil = Var(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        initialize=1,
        units=pyunits.dimensionless,
        bounds=(0, None),
        doc="""Reduced concentration of CO2
                at chemical equilibrium with
                interfacial concentrations,
                Driving force term in which
                Absorption implies conc_CO2_bulk < 1 and 
                Desorption implies conc_CO2_bulk > 1 """,
    )

    @blk.Constraint(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        doc="""Dimensionless concentration of CO2""",
    )
    def red_conc_CO2_bulk_eqn(b, t, x):
        if x == b.liquid_phase.length_domain.last():
            return Constraint.Skip
        else:
            zf = b.liquid_phase.length_domain.next(x)
            P_CO2 = pyunits.convert(
                b.pressure_equil[t, zf, "CO2"],
                to_units=lunits("pressure"),
            )
            return b.red_conc_CO2_bulk[t, x] == (
                b.liquid_phase.properties[t, x].henry["Liq", "CO2"]
                * b.liquid_phase.properties[t, x].conc_mol_phase_comp_true["Liq", "CO2"]
                / P_CO2
            )

    @blk.Expression(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        doc="""Instantaneous Enhancement factor""",
    )
    def instant_E_minus_one(b, t, x):
        if x == b.liquid_phase.length_domain.last():
            return Expression.Skip
        else:
            zf = b.liquid_phase.length_domain.next(x)
            P_CO2 = pyunits.convert(
                b.pressure_equil[t, zf, "CO2"],
                to_units=lunits("pressure"),
            )
            return (
                b.liquid_phase.properties[t, x].diffus_phase_comp["Liq", "MEA"]
                * b.liquid_phase.properties[t, x].conc_mol_phase_comp_true["Liq", "MEA"]
                * b.liquid_phase.properties[t, x].henry["Liq", "CO2"]
                / (
                    2
                    * b.liquid_phase.properties[t, x].diffus_phase_comp["Liq", "CO2"]
                    * P_CO2
                )

            )

    # ======================================================================
    # Enhancement factor model
    # Reference: Jozsef Gaspar,Philip Loldrup Fosbol, (2015)

    blk.red_conc_interface_MEA = Var(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        bounds=(0, None),
        initialize=1,
        units=pyunits.dimensionless,
        doc="""Dimensionless concentration of MEA
                                    at interface """,
    )

    @blk.Expression(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        doc="Dimensionless concentration of MEACOO-",
    )
    def red_conc_interface_MEACOO(b, t, x):
        if x == b.liquid_phase.length_domain.last():
            return Expression.Skip
        else:
            return 1 + (
                b.liquid_phase.properties[t, x].diffus_phase_comp_true["Liq", "MEA"]
                * b.liquid_phase.properties[t, x].conc_mol_phase_comp_true["Liq", "MEA"]
            ) * (1 - b.red_conc_interface_MEA[t, x]) / (
                2
                * b.liquid_phase.properties[t, x].diffus_phase_comp_true[
                    "Liq", "MEACOO_-"
                ]
                * b.liquid_phase.properties[t, x].conc_mol_phase_comp_true[
                    "Liq", "MEACOO_-"
                ]
            )

    @blk.Expression(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        doc="Dimensionless concentration of MEA+",
    )
    def red_conc_interface_MEAH(b, t, x):
        if x == b.liquid_phase.length_domain.last():
            return Expression.Skip
        else:
            return 1 + (
                b.liquid_phase.properties[t, x].diffus_phase_comp_true["Liq", "MEA"]
                * b.liquid_phase.properties[t, x].conc_mol_phase_comp_true["Liq", "MEA"]
            ) * (1 - b.red_conc_interface_MEA[t, x]) / (
                2
                * b.liquid_phase.properties[t, x].diffus_phase_comp_true["Liq", "MEA_+"]
                * b.liquid_phase.properties[t, x].conc_mol_phase_comp_true[
                    "Liq", "MEA_+"
                ]
            )

    @blk.Constraint(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        doc="""Constraint for dimensionless concentration of CO2
                              at equilibrium with the bulk """,
    )
    def red_conc_CO2_equil_eqn(b, t, x):
        if x == b.liquid_phase.length_domain.last():
            return Constraint.Skip
        else:
            return (
                b.red_conc_CO2_equil[t, x] * b.red_conc_interface_MEA[t, x] ** 2
                == b.red_conc_CO2_bulk[t, x]
                * b.red_conc_interface_MEAH[t, x]
                * b.red_conc_interface_MEACOO[t, x]
            )

    @blk.Constraint(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        doc="Enhancement factor - function of Hatta number",
    )
    def enhancement_factor_eqn1(b, t, x):
        if x == b.liquid_phase.length_domain.last():
            return Constraint.Skip
        else:
            return b.omega[t, x] == (
                b.hatta_number[t, x]
                * sqrt(b.red_conc_interface_MEA[t, x])
                * (1 - b.red_conc_CO2_equil[t, x])
            )

    @blk.Constraint(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        doc="Enhancement factor - function of instantaneous enhancement factor",
    )
    def enhancement_factor_eqn2(b, t, x):
        if x == b.liquid_phase.length_domain.last():
            return Constraint.Skip
        else:
            return b.omega[t, x] == (
                1 - b.red_conc_CO2_bulk[t, x] +
                + b.instant_E_minus_one[t, x]
                * (1 - b.red_conc_interface_MEA[t, x])
            )

    enhancement_factor_vars = [
        blk.omega,
        blk.red_conc_CO2_bulk,
        blk.red_conc_CO2_equil,
        blk.red_conc_interface_MEA,
        # Also entangled is blk.pressure_equil, but we don't want that fixed
    ]

    enhancement_factor_constraints = [
        blk.red_conc_CO2_bulk_eqn,
        blk.red_conc_CO2_equil_eqn,
        blk.enhancement_factor_eqn1,
        blk.enhancement_factor_eqn2,
        # Also entangled is blk.pressure_at_interface, but we don't want that deactivated
    ]

    return enhancement_factor_vars, enhancement_factor_constraints

def enhancement_factor_initial_guess(blk):
    def diffus_ratio(b, t, x, j):
        # When evaluated for CO2, we get instant_E_hat_minus_one
        if x == b.liquid_phase.length_domain.last():
            return RuntimeError
        props = b.liquid_phase.properties[t,x]
        return value(
            props.diffus_phase_comp_true["Liq","MEA"]
            * props.conc_mol_phase_comp_true["Liq","MEA"]
            / (
                2 * props.diffus_phase_comp_true["Liq", j]
                * props.conc_mol_phase_comp_true["Liq", j]
            )
        )
    for t in blk.flowsheet().time:
        for x in blk.liquid_phase.length_domain:
            if x == blk.liquid_phase.length_domain.last():
                continue
            zf = blk.liquid_phase.length_domain.next(x)
            R_plus = diffus_ratio(blk, t, x, "MEA_+")
            R_minus = diffus_ratio(blk, t, x, "MEACOO_-")
            Estar_hat_minus_one = diffus_ratio(blk, t, x, "CO2")
            Ha = value(blk.hatta_number[t, x])
            Ehat = 1 + (Ha - 1) /(Ha*(R_plus+R_minus+2)/Estar_hat_minus_one +1)
            He = blk.liquid_phase.properties[t, x].henry["Liq","CO2"]
            C_CO2_bulk = blk.liquid_phase.properties[t, x].conc_mol_phase_comp_true["Liq","CO2"]
            P_CO2_bulk = blk.vapor_phase.properties[t, zf].fug_phase_comp["Vap","CO2"]
            kL = blk.mass_transfer_coeff_liq[t, x, "CO2"]
            kV = blk.mass_transfer_coeff_vap[t, zf, "CO2"]
            C_CO2_interface = C_CO2_bulk + kV * ( 
                P_CO2_bulk - He * C_CO2_bulk
            ) / (Ehat*kL + kV*He)

            Y_b_CO2 = value(C_CO2_bulk/C_CO2_interface)
            blk.red_conc_CO2_bulk[t, x].set_value(Y_b_CO2)
            blk.omega[t, x].set_value(Ehat*(1 - Y_b_CO2))
            Y_i_MEA = value(
                1
                - (Ehat*(1 - Y_b_CO2) - 1)
                / Estar_hat_minus_one
            )
            blk.red_conc_interface_MEA[t, x].set_value(Y_i_MEA)
            calculate_variable_from_constraint(blk.red_conc_CO2_equil[t, x], blk.red_conc_CO2_equil_eqn[t, x])

def initialize_enhancement_factor_model(
    blk,
    state_args=None,
    outlvl=idaeslog.NOTSET,
    optarg=None,
    solver=None,
):
    # Set up logger for initialization and solve
    init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
    solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="unit")

    enhancement_factor_initial_guess(blk)

    # Set solver options
    if optarg is None:
        optarg = {}

    solver_obj = get_solver(solver, optarg)

    long_var_list = []
    long_eqn_list = []

    for t in blk.flowsheet().time:
        for x in blk.liquid_phase.length_domain:
            if x == blk.liquid_phase.length_domain.last():
                continue
            zf = blk.liquid_phase.length_domain.next(x)
            entangled_vars = [
                blk.omega[t, x],
                blk.omega_slack[t, x],
                blk.red_conc_CO2_bulk[t, x],
                blk.red_conc_CO2_equil[t, x],
                blk.red_conc_interface_MEA[t, x],
                blk.pressure_equil[t, zf, "CO2"],
            ]
            entangled_eqns = [
                blk.omega_slack_eqn[t, x],
                blk.red_conc_CO2_bulk_eqn[t, x],
                blk.red_conc_CO2_equil_eqn[t, x],
                blk.enhancement_factor_eqn1[t, x],
                blk.enhancement_factor_eqn2[t, x],
                blk.pressure_at_interface[t, zf, "CO2"],
            ]
            long_var_list += entangled_vars
            long_eqn_list += entangled_eqns

            tmp_blk = create_subsystem_block(entangled_eqns, entangled_vars)
            _sub_problem_scaling_suffix(blk, tmp_blk)
            flags = _fix_vars([var for var in tmp_blk.input_vars.values()])
            assert degrees_of_freedom(tmp_blk) == 0
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = solver_obj.solve(tmp_blk, tee=slc.tee, symbolic_solver_labels=True)
            assert_optimal_termination(res)
            _restore_fixedness(flags)

    # import pdb; pdb.set_trace()
    tmp_blk = create_subsystem_block(long_eqn_list, long_var_list)
    _sub_problem_scaling_suffix(blk, tmp_blk)
    flags = _fix_vars([var for var in tmp_blk.input_vars.values()])
    assert degrees_of_freedom(tmp_blk) == 0
    with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
        res = solver_obj.solve(tmp_blk, tee=slc.tee, symbolic_solver_labels=True)
    assert_optimal_termination(res)

    _restore_fixedness(flags)
