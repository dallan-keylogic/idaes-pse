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
General purpose mixer block for IDAES models
"""
from enum import Enum

from pyomo.environ import (
    Block,
    check_optimal_termination,
    Constraint,
    Param,
    PositiveReals,
    Reals,
    RangeSet,
    Var,
    value,
)
from pyomo.common.config import ConfigBlock, ConfigValue, In, ListOf, Bool

from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
    MaterialBalanceType,
    MaterialFlowBasis,
)
from idaes.core.util.config import (
    is_physical_parameter_block,
    is_state_block,
)
from idaes.core.util.exceptions import (
    BurntToast,
    ConfigurationError,
    PropertyNotSupportedError,
    InitializationError,
)
from idaes.core.util.math import smooth_min
from idaes.core.util.tables import create_stream_table_dataframe
import idaes.core.util.scaling as iscale
from idaes.core.solvers import get_solver
from idaes.core.initialization import ModularInitializerBase
from idaes.models.unit_models.mixer import MixingType, MomentumMixingType
from idaes.models_extra.column_models.properties import ModularPropertiesInherentReactionsInitializer

import idaes.logger as idaeslog

__author__ = "Andrew Lee"


# Set up logger
_log = idaeslog.getLogger(__name__)


class TranslatingMixerInitializer(ModularInitializerBase):
    """
    Hierarchical Initializer for Mixer blocks.

    """

    def initialization_routine(
        self,
        model: Block,
    ):
        """
        Initialization routine for Mixer Blocks.

        This routine starts by initializing each of the inlet streams, then uses those
        results to estimate the outlet state before solving the full model.

        Args:
            model: model to be initialized

        Returns:
            Pyomo solver status object
        """
        init_log = idaeslog.getInitLogger(
            model.name, self.get_output_level(), tag="unit"
        )
        solve_log = idaeslog.getSolveLogger(
            model.name, self.get_output_level(), tag="unit"
        )

        # Create solver
        solver = self._get_solver()

        # Initialize inlet state blocks
        inlet_list = model.create_inlet_list()
        i_block_list = []
        for i in inlet_list:
            i_block = getattr(model, i + "_state")
            i_block_list.append(i_block)

            # Get initializer for inlet
            iinit = self.get_submodel_initializer(i_block)

            iinit.initialize(i_block)

        # Initialize mixed state block
        if model.config.mixed_state_block is None:
            mblock = model.mixed_state
        else:
            mblock = model.config.mixed_state_block

        # Calculate initial guesses for mixed stream state
        for t in model.flowsheet().time:
            # Iterate over state vars as defined by property package
            s_vars = mblock[t].define_state_vars()
            for s in s_vars:
                for k in s_vars[s]:
                    # If fixed, use current value
                    # otherwise calculate guess from mixed state
                    if not s_vars[s][k].fixed:
                        if s == "pressure":
                            # If pressure, use minimum as initial guess
                            mblock[t].pressure.set_value(
                                min(
                                    value(i_block_list[i][t].pressure)
                                    for i in range(len(i_block_list))
                                )
                            )
                        elif "flow" in s:
                            if "flow_comp" in s:
                                validation_sets = [i_block_list[i][t].component_list for i in range(len(i_block_list))]
                            elif "flow_phase_comp" in s:
                                validation_sets = [i_block_list[i][t].phase_component_set for i in range(len(i_block_list))]
                            else:
                                # Otherwise s is unindexed, right..? FIXME
                                validation_sets = [{None} for i in range(len(i_block_list))]
                            # If a "flow" variable (i.e. extensive), sum inlets
                            s_vars[s][k].set_value(
                                sum(
                                    value(
                                        getattr(i_block_list[i][t], s)[k]
                                        # TODO the attribute will always be generated, right?
                                    )
                                    for i in range(len(i_block_list))
                                    if k in validation_sets[i]
                                )
                                )
                        elif "mole_frac" in s:
                            if "mole_frac_comp" in s:
                                validation_sets = [i_block_list[i][t].component_list for i in range(len(i_block_list))]
                                flow_mol = max(
                                    sum(value(i_block_list[i][t].flow_mol) for i in range(len(i_block_list))),
                                    1e-16 #Guard against zero flows
                                )
                                flow_mol_comp = sum(
                                    value(
                                        i_block_list[i][t].flow_mol_comp[k]
                                        # TODO the attribute will always be generated, right?
                                    )
                                    for i in range(len(i_block_list))
                                    if k in validation_sets[i]
                                )
                                s_vars[s][k].set_value(flow_mol_comp/flow_mol)
                            
                            elif "mole_frac_phase_comp" in s:
                                phase_sets = [i_block_list[i][t].phase_set for i in range(len(i_block_list))]
                                phase_comp_sets = [i_block_list[i][t].phase_component_set for i in range(len(i_block_list))]
                                p = k=[0]
                                flow_mol_phase = max(
                                    sum(
                                        value(i_block_list[i][t].flow_mol_phase[p])
                                        for i in range(len(i_block_list))
                                        if p in phase_sets[i]
                                    ),
                                    1e-16 #Guard against zero flows
                                )
                                flow_mol_phase_comp = sum(
                                    value(
                                        i_block_list[i][t].flow_mol_phase_comp[k]
                                        # TODO the attribute will always be generated, right?
                                    )
                                    for i in range(len(i_block_list))
                                    if k in phase_comp_sets[i]
                                )
                                s_vars[s][k].set_value(flow_mol_phase_comp/flow_mol_phase)

                        else:
                            # Otherwise use average of inlets
                            # TODO need to add a step to iterate through and generate these values
                            # What is going to be here besides temperature or molar enthalpy? Volume?
                            s_vars[s][k].set_value(
                                sum(
                                    value(
                                        getattr(i_block_list[i][t], s)[k]
                                    )
                                    for i in range(len(i_block_list))
                                )
                                / len(i_block_list)
                            )

        # Get initializer for mixed block
        minit = self.get_submodel_initializer(mblock)
        minit.initialize(mblock)

        res = None

        if model.config.mixed_state_block is None:
            if (
                hasattr(model, "pressure_equality_constraints")
                and model.pressure_equality_constraints.active is True
            ):
                model.pressure_equality_constraints.deactivate()
                for t in model.flowsheet().time:
                    sys_press = getattr(model, model.create_inlet_list()[0] + "_state")[
                        t
                    ].pressure
                    model.mixed_state[t].pressure.fix(sys_press.value)
                with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                    res = solver.solve(model, tee=slc.tee)
                model.pressure_equality_constraints.activate()
                for t in model.flowsheet().time:
                    model.mixed_state[t].pressure.unfix()
            else:
                # import pdb; pdb.set_trace()
                with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                    res = solver.solve(model, tee=slc.tee)

            init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))
        else:
            init_log.info("Initialization Complete.")

        return res


@declare_process_block_class("TranslatingMixer")
class TranslatingMixerData(UnitModelBlockData):
    """
    TODO rewrite
    This is a general purpose model for a Mixer block with the IDAES modeling
    framework. This block can be used either as a stand-alone Mixer unit
    operation, or as a sub-model within another unit operation.

    This model creates a number of StateBlocks to represent the incoming
    streams, then writes a set of phase-component material balances, an
    overall enthalpy balance and a momentum balance (2 options) linked to a
    mixed-state StateBlock. The mixed-state StateBlock can either be specified
    by the user (allowing use as a sub-model), or created by the Mixer.

    When being used as a sub-model, Mixer should only be used when a set
    of new StateBlocks are required for the streams to be mixed. It should not
    be used to mix streams from multiple ControlVolumes in a single unit model -
    in these cases the unit model developer should write their own mixing
    equations.
    """

    default_initializer = TranslatingMixerInitializer

    CONFIG = ConfigBlock()
    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False",
            doc="""Indicates whether this model will be dynamic or not,
**default** = False. Mixer blocks are always steady-state.""",
        ),
    )
    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False",
            doc="""Mixer blocks do not contain holdup, thus this must be
False.""",
        ),
    )
    CONFIG.declare(
        "inlet_dict",
        ConfigValue(
            # domain=ListOf(str),
            description="List of inlet names",
            doc="""A dictionary with inlet names as keys and dictionaries as values.
                The dictionaries should all have "property_package" as a key and a 
                property parameter object as value. Optionally, they can contain 
                "property_package_args" as a key and a ConfigBlock with arguments to
                be passed to the property block as values.
                TODO enthalpy offset?
                """,
        ),
    )
    CONFIG.declare(
        "mixed_state_property_package",
        ConfigValue(
            # default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for mixed state",
            doc="""Property parameter object used to define property
            calculations,
            **Valid values:** {
            **PropertyParameterObject** - a PropertyParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "mixed_state_property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing mixed state property package",
            doc="""A ConfigBlock with arguments to be passed to the mixed state property
                block and used when constructing it,
                **default** - None.
                **Valid values:** {
                see property package for documentation.}""",
        ),
    )
    CONFIG.declare(
        "material_balance_type",
        ConfigValue(
            default=MaterialBalanceType.useDefault,
            domain=In(MaterialBalanceType),
            description="Material balance construction flag",
            doc="""Indicates what type of mass balance should be constructed,
**default** - MaterialBalanceType.useDefault.
**Valid values:** {
**MaterialBalanceType.useDefault - refer to property package for default
balance type
**MaterialBalanceType.none** - exclude material balances,
**MaterialBalanceType.componentPhase** - use phase component balances,
**MaterialBalanceType.componentTotal** - use total component balances,
**MaterialBalanceType.elementTotal** - use total element balances,
**MaterialBalanceType.total** - use total material balance.}""",
        ),
    )
    CONFIG.declare(
        "has_phase_equilibrium",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Calculate phase equilibrium in mixed stream",
            doc="""Argument indicating whether phase equilibrium should be
calculated for the resulting mixed stream,
**default** - False.
**Valid values:** {
**True** - calculate phase equilibrium in mixed stream,
**False** - do not calculate equilibrium in mixed stream.}""",
        ),
    )
    CONFIG.declare(
        "energy_mixing_type",
        ConfigValue(
            default=MixingType.extensive,
            domain=MixingType,
            description="Method to use when mixing energy flows",
            doc="""Argument indicating what method to use when mixing energy
flows of incoming streams,
**default** - MixingType.extensive.
**Valid values:** {
**MixingType.none** - do not include energy mixing equations,
**MixingType.extensive** - mix total enthalpy flows of each phase.}""",
        ),
    )
    CONFIG.declare(
        "momentum_mixing_type",
        ConfigValue(
            default=MomentumMixingType.minimize,
            domain=MomentumMixingType,
            description="Method to use when mixing momentum/pressure",
            doc="""Argument indicating what method to use when mixing momentum/
pressure of incoming streams,
**default** - MomentumMixingType.minimize.
**Valid values:** {
**MomentumMixingType.none** - do not include momentum mixing equations,
**MomentumMixingType.minimize** - mixed stream has pressure equal to the
minimimum pressure of the incoming streams (uses smoothMin operator),
**MomentumMixingType.equality** - enforces equality of pressure in mixed and
all incoming streams.,
**MomentumMixingType.minimize_and_equality** - add constraints for pressure
equal to the minimum pressure of the inlets and constraints for equality of
pressure in mixed and all incoming streams. When the model is initially built,
the equality constraints are deactivated.  This option is useful for switching
between flow and pressure driven simulations.}""",
        ),
    )
    CONFIG.declare(
        "mixed_state_block",
        ConfigValue(
            default=None,
            domain=is_state_block,
            description="Existing StateBlock to use as mixed stream",
            doc="""An existing state block to use as the outlet stream from the
Mixer block,
**default** - None.
**Valid values:** {
**None** - create a new StateBlock for the mixed stream,
**StateBlock** - a StateBock to use as the destination for the mixed stream.}
""",
        ),
    )
#     CONFIG.declare(
#         "construct_ports",
#         ConfigValue(
#             default=True,
#             domain=Bool,
#             description="Construct inlet and outlet Port objects",
#             doc="""Argument indicating whether model should construct Port
# objects linked to all inlet states and the mixed state,
# **default** - True.
# **Valid values:** {
# **True** - construct Ports for all states,
# **False** - do not construct Ports.""",
#         ),
#     )

    def build(self):
        """
        General build method for MixerData. This method calls a number
        of sub-methods which automate the construction of expected attributes
        of unit models.

        Inheriting models should call `super().build`.

        Args:
            None

        Returns:
            None
        """
        # Call super.build()
        super(TranslatingMixerData, self).build()

        # Create list of inlet names
        inlet_list = self.create_inlet_list()

        # Build StateBlocks
        inlet_blocks = self.add_inlet_state_blocks()

        if self.config.mixed_state_block is None:
            mixed_block = self.add_mixed_state_block()
        else:
            mixed_block = self.get_mixed_state_block()

        mb_type = self.config.material_balance_type
        if mb_type == MaterialBalanceType.useDefault:
            t_ref = self.flowsheet().time.first()
            mb_type = mixed_block[t_ref].default_material_balance_type()

        if mb_type != MaterialBalanceType.none:
            self.add_material_mixing_equations(
                inlet_blocks=inlet_blocks, mixed_block=mixed_block, mb_type=mb_type
            )
        else:
            raise BurntToast(
                "{} received unrecognised value for "
                "material_mixing_type argument. This "
                "should not occur, so please contact "
                "the IDAES developers with this bug.".format(self.name)
            )

        if self.config.energy_mixing_type == MixingType.extensive:
            self.add_energy_mixing_equations(
                inlet_blocks=inlet_blocks, mixed_block=mixed_block
            )
        elif self.config.energy_mixing_type == MixingType.none:
            pass
        else:
            raise ConfigurationError(
                "{} received unrecognised value for "
                "material_mixing_type argument. This "
                "should not occur, so please contact "
                "the IDAES developers with this bug.".format(self.name)
            )

        # Add to try/expect to catch cases where pressure is not supported
        # by properties.
        try:
            if self.config.momentum_mixing_type == MomentumMixingType.minimize:
                self.add_pressure_minimization_equations(
                    inlet_blocks=inlet_blocks, mixed_block=mixed_block
                )
            elif self.config.momentum_mixing_type == MomentumMixingType.equality:
                self.add_pressure_equality_equations(
                    inlet_blocks=inlet_blocks, mixed_block=mixed_block
                )
            elif (
                self.config.momentum_mixing_type
                == MomentumMixingType.minimize_and_equality
            ):
                self.add_pressure_minimization_equations(
                    inlet_blocks=inlet_blocks, mixed_block=mixed_block
                )
                self.add_pressure_equality_equations(
                    inlet_blocks=inlet_blocks, mixed_block=mixed_block
                )
                self.pressure_equality_constraints.deactivate()
            elif self.config.momentum_mixing_type == MomentumMixingType.none:
                pass
            else:
                raise ConfigurationError(
                    "{} received unrecognised value for "
                    "momentum_mixing_type argument. This "
                    "should not occur, so please contact "
                    "the IDAES developers with this bug.".format(self.name)
                )
        except PropertyNotSupportedError:
            raise PropertyNotSupportedError(
                "{} The property package supplied for this unit does not "
                "appear to support pressure, which is required for momentum "
                "mixing. Please set momentum_mixing_type to "
                "MomentumMixingType.none or provide a property package which "
                "supports pressure.".format(self.name)
            )

        self.add_port_objects(inlet_list, inlet_blocks, mixed_block)

    def create_inlet_list(self):
        """
        Create list of inlet stream names based on config arguments.

        Returns:
            list of strings
        """
        # Create a list of names for inlet StateBlocks
        if self.config.inlet_dict is not None:
            inlet_list = [key for key in self.config.inlet_dict.keys()]
        else:
            raise ConfigurationError(
                f"{self.name} Mixer not provided with inlet dict."
            )

        return inlet_list

    def add_inlet_state_blocks(self):
        """
        Construct StateBlocks for all inlet streams.

        Args:
            None

        Returns:
            list of StateBlocks
        """
        # Setup StateBlock argument dict
        inlet_dict = self.config.inlet_dict
        # Create empty list to hold StateBlocks for return
        inlet_blocks = []
        for inlet_name, config_dict in inlet_dict.items():
            inlet_properties = config_dict["property_package"]
            try:
                arg_dict = config_dict["property_package_args"]
            except KeyError:
                arg_dict = {}
            tmp_dict = dict(**arg_dict)
            tmp_dict["has_phase_equilibrium"] = False
            tmp_dict["defined_state"] = True
            inlet_state = inlet_properties.build_state_block(
                self.flowsheet().time, doc=f"Material properties at inlet {inlet_name}", **tmp_dict
            )
            setattr(self, inlet_name + "_state", inlet_state)
            inlet_blocks.append(inlet_state)

        return inlet_blocks

    def add_mixed_state_block(self):
        """
        Constructs StateBlock to represent mixed stream.

        Returns:
            New StateBlock object
        """
        # Setup StateBlock argument dict
        tmp_dict = dict(**self.config.mixed_state_property_package_args)
        tmp_dict["has_phase_equilibrium"] = self.config.has_phase_equilibrium
        tmp_dict["defined_state"] = False

        self.mixed_state = self.config.mixed_state_property_package.build_state_block(
            self.flowsheet().time, doc="Material properties of mixed stream", **tmp_dict
        )

        return self.mixed_state

    def get_mixed_state_block(self):
        """
        Validates StateBlock provided in user arguments for mixed stream.

        Returns:
            The user-provided StateBlock or an Exception
        """
        # Sanity check to make sure method is not called when arg missing
        if self.config.mixed_state_block is None:
            raise BurntToast(
                "{} get_mixed_state_block method called when "
                "mixed_state_block argument is None. This should "
                "not happen.".format(self.name)
            )

        # Check that the user-provided StateBlock uses the same prop pack
        # if (
        #     self.config.mixed_state_block[
        #         self.flowsheet().time.first()
        #     ].config.parameters
        #     != self.config.property_package
        # ):
        #     raise ConfigurationError(
        #         "{} StateBlock provided in mixed_state_block argument "
        #         "does not come from the same property package as "
        #         "provided in the property_package argument. All "
        #         "StateBlocks within a Mixer must use the same "
        #         "property package.".format(self.name)
        #     )

        return self.config.mixed_state_block

    def add_material_mixing_equations(self, inlet_blocks, mixed_block, mb_type):
        """
        Add material mixing equations.
        """
        pp = self.config.mixed_state_property_package
        # Get phase component list(s)
        pc_set = mixed_block.phase_component_set

        # Get units metadata
        units = pp.get_metadata()

        flow_basis = mixed_block[
            self.flowsheet().time.first()
        ].get_material_flow_basis()
        if flow_basis == MaterialFlowBasis.molar:
            flow_units = units.get_derived_units("flow_mole")
        elif flow_basis == MaterialFlowBasis.mass:
            flow_units = units.get_derived_units("flow_mass")
        else:
            # Let this pass for now with no units
            flow_units = None

        if mixed_block.include_inherent_reactions:
            if mb_type == MaterialBalanceType.total:
                raise ConfigurationError(
                    "Cannot do total flow mixing with inherent reaction; "
                    "problem is under-constrained. Please use a different "
                    "mixing type."
                )

            # Add extents of reaction and stoichiometric constraints
            # TODO is this compatible with mass flows?
            self.inherent_reaction_extent = Var(
                self.flowsheet().time,
                mixed_block.params.inherent_reaction_idx,
                domain=Reals,
                initialize=0.0,
                doc="Extent of inherent reactions in outlet",
                units=flow_units,
            )

            self.inherent_reaction_generation = Var(
                self.flowsheet().time,
                pc_set,
                domain=Reals,
                initialize=0.0,
                doc="Generation due to inherent reactions in outlet",
                units=flow_units,
            )

            @self.Constraint(
                self.flowsheet().time,
                pc_set,
            )
            def inherent_reaction_constraint(b, t, p, j):
                if (p, j) in pc_set:
                    return b.inherent_reaction_generation[t, p, j] == (
                        sum(
                            mixed_block[t].params.inherent_reaction_stoichiometry[
                                r, p, j
                            ]
                            * self.inherent_reaction_extent[t, r]
                            for r in mixed_block[t].params.inherent_reaction_idx
                        )
                    )
                return Constraint.Skip
        # TODO add unit conversions
        if mb_type == MaterialBalanceType.componentPhase:
            # Create equilibrium generation term and constraints if required
            if self.config.has_phase_equilibrium is True:
                try:
                    self.phase_equilibrium_generation = Var(
                        self.flowsheet().time,
                        pp.phase_equilibrium_idx,
                        domain=Reals,
                        doc="Amount of generation in unit by phase equilibria",
                        units=flow_units,
                    )
                except AttributeError:
                    raise PropertyNotSupportedError(
                        "{} Property package does not contain a list of phase "
                        "equilibrium reactions (phase_equilibrium_idx), "
                        "thus does not support phase equilibrium.".format(self.name)
                    )

            # Write phase-component balances
            @self.Constraint(
                self.flowsheet().time,
                pc_set,
                doc="Material mixing equations",
            )
            def material_mixing_equations(b, t, p, j):
                # inlet_sum = 0
                # for i in range(len(inlet_blocks):
                #     if (p, j) in inlet_blocks[i].phase_component_set:
                #         inlet_sum += inlet_blocks[i][t].get_material_flow_terms(p, j)

                rhs = sum(
                    inlet_blocks[i][t].get_material_flow_terms(p, j)
                    for i in range(len(inlet_blocks))
                    if (p, j) in inlet_blocks[i].phase_component_set
                ) - mixed_block[t].get_material_flow_terms(p, j)

                if self.config.has_phase_equilibrium:
                    rhs += sum(
                        b.phase_equilibrium_generation[t, r]
                        for r in pp.phase_equilibrium_idx
                        if pp.phase_equilibrium_list[r][0] == j
                        and pp.phase_equilibrium_list[r][1][0] == p
                    ) - sum(
                        b.phase_equilibrium_generation[t, r]
                        for r in pp.phase_equilibrium_idx
                        if pp.phase_equilibrium_list[r][0] == j
                        and pp.phase_equilibrium_list[r][1][1] == p
                    )

                if mixed_block.include_inherent_reactions:
                    rhs += b.inherent_reaction_generation[t, p, j]

                return 0 == rhs

        elif mb_type == MaterialBalanceType.componentTotal:
            # Write phase-component balances
            @self.Constraint(
                self.flowsheet().time,
                mixed_block.component_list,
                doc="Material mixing equations",
            )
            def material_mixing_equations(b, t, j):
                rhs = sum(
                    sum(
                        inlet_blocks[i][t].get_material_flow_terms(p, j)
                        for i in range(len(inlet_blocks))
                        if j in inlet_blocks[i].component_list
                    )
                    - mixed_block[t].get_material_flow_terms(p, j)
                    for p in mixed_block.phase_list
                    if (p, j) in pc_set
                )

                if mixed_block.include_inherent_reactions:
                    rhs += sum(
                        b.inherent_reaction_generation[t, p, j]
                        for p in mixed_block.phase_list
                        if (p, j) in pc_set
                    )

                return 0 == rhs

        elif mb_type == MaterialBalanceType.total:
            # Write phase-component balances
            @self.Constraint(self.flowsheet().time, doc="Material mixing equations")
            def material_mixing_equations(b, t):
                rhs = sum(
                    sum(
                        sum(
                            inlet_blocks[i][t].get_material_flow_terms(p, j)
                            for i in range(len(inlet_blocks))
                            if (p, j) in inlet_blocks[i].phase_component_set
                        )
                        - mixed_block[t].get_material_flow_terms(p, j)
                        for j in mixed_block.component_list
                        if (p, j) in pc_set
                    )
                    for p in mixed_block.phase_list
                )

                return 0 == rhs

        elif mb_type == MaterialBalanceType.elementTotal:
            raise ConfigurationError(
                "{} Mixers do not support elemental "
                "material balances.".format(self.name)
            )
        elif mb_type == MaterialBalanceType.none:
            pass
        else:
            raise BurntToast(
                "{} Mixer received unrecognised value for "
                "material_balance_type. This should not happen, "
                "please report this bug to the IDAES developers.".format(self.name)
            )

    def add_energy_mixing_equations(self, inlet_blocks, mixed_block):
        """
        Add energy mixing equations (total enthalpy balance).
        """

        @self.Constraint(self.flowsheet().time, doc="Energy balances")
        def enthalpy_mixing_equations(b, t):
            return 0 == (
                sum(
                    sum(
                        inlet_blocks[i][t].get_enthalpy_flow_terms(p)
                        for p in mixed_block.phase_list
                        if p in inlet_blocks[i].phase_list
                    )
                    for i in range(len(inlet_blocks))
                )
                - sum(
                    mixed_block[t].get_enthalpy_flow_terms(p)
                    for p in mixed_block.phase_list
                )
            )

    def add_pressure_minimization_equations(self, inlet_blocks, mixed_block):
        """
        Add pressure minimization equations. This is done by sequential
        comparisons of each inlet to the minimum pressure so far, using
        the IDAES smooth minimum function.
        """
        if not hasattr(self, "inlet_idx"):
            self.inlet_idx = RangeSet(len(inlet_blocks))

        # Get units metadata
        units = mixed_block.params.get_metadata()

        # Add variables
        self.minimum_pressure = Var(
            self.flowsheet().time,
            self.inlet_idx,
            doc="Variable for calculating minimum inlet pressure",
            units=units.get_derived_units("pressure"),
        )

        self.eps_pressure = Param(
            mutable=True,
            initialize=1e-3,
            domain=PositiveReals,
            doc="Smoothing term for minimum inlet pressure",
            units=units.get_derived_units("pressure"),
        )

        # Calculate minimum inlet pressure
        @self.Constraint(
            self.flowsheet().time,
            self.inlet_idx,
            doc="Calculation for minimum inlet pressure",
        )
        def minimum_pressure_constraint(b, t, i):
            if i == self.inlet_idx.first():
                return self.minimum_pressure[t, i] == (inlet_blocks[i - 1][t].pressure)
            else:
                return self.minimum_pressure[t, i] == (
                    smooth_min(
                        self.minimum_pressure[t, i - 1],
                        inlet_blocks[i - 1][t].pressure,
                        self.eps_pressure,
                    )
                )

        # Set inlet pressure to minimum pressure
        @self.Constraint(self.flowsheet().time, doc="Link pressure to control volume")
        def mixture_pressure(b, t):
            return mixed_block[t].pressure == (
                self.minimum_pressure[t, self.inlet_idx.last()]
            )

    def add_pressure_equality_equations(self, inlet_blocks, mixed_block):
        """
        Add pressure equality equations. Note that this writes a number of
        constraints equal to the number of inlets, enforcing equality between
        all inlets and the mixed stream.
        """
        if not hasattr(self, "inlet_idx"):
            self.inlet_idx = RangeSet(len(inlet_blocks))

        # Create equality constraints
        @self.Constraint(
            self.flowsheet().time,
            self.inlet_idx,
            doc="Calculation for minimum inlet pressure",
        )
        def pressure_equality_constraints(b, t, i):
            return mixed_block[t].pressure == inlet_blocks[i - 1][t].pressure

    def add_port_objects(self, inlet_list, inlet_blocks, mixed_block):
        """
        Adds Port objects if required.

        Args:
            a list of inlet StateBlock objects
            a mixed state StateBlock object

        Returns:
            None
        """
        # if self.config.construct_ports is True:
        # Add ports
        for p in inlet_list:
            i_state = getattr(self, p + "_state")
            self.add_port(name=p, block=i_state, doc="Inlet Port")
        self.add_port(name="outlet", block=mixed_block, doc="Outlet Port")

    def model_check(blk):
        """
        This method executes the model_check methods on the associated state
        blocks (if they exist). This method is generally called by a unit model
        as part of the unit's model_check method.

        Args:
            None

        Returns:
            None
        """
        # TODO add check about significant amount of components being dropped
        # because they're present in inlet blocks but not the mixed block
        # (and thus not getting written into any material balance constraint)
        pass
        # Try property block model check
        # for t in blk.flowsheet().time:
        #     try:
        #         inlet_list = blk.create_inlet_list()
        #         for i in inlet_list:
        #             i_block = getattr(blk, i + "_state")
        #             i_block[t].model_check()
        #     except AttributeError:
        #         _log.warning(
        #             "{} Mixer inlet property block has no model "
        #             "checks. To correct this, add a model_check "
        #             "method to the associated StateBlock class.".format(blk.name)
        #         )
        #     try:
        #         if blk.config.mixed_state_block is None:
        #             blk.mixed_state[t].model_check()
        #         else:
        #             blk.config.mixed_state_block.model_check()
        #     except AttributeError:
        #         _log.warning(
        #             "{} Mixer outlet property block has no "
        #             "model checks. To correct this, add a "
        #             "model_check method to the associated "
        #             "StateBlock class.".format(blk.name)
        #         )

    def use_minimum_inlet_pressure_constraint(self):
        """Activate the mixer pressure = minimum inlet pressure constraint and
        deactivate the mixer pressure and all inlet pressures are equal
        constraints. This should only be used when momentum_mixing_type ==
        MomentumMixingType.minimize_and_equality.
        """
        if self.config.momentum_mixing_type != MomentumMixingType.minimize_and_equality:
            _log.warning(
                """use_minimum_inlet_pressure_constraint() can only be used
                when momentum_mixing_type ==
                MomentumMixingType.minimize_and_equality"""
            )
            return
        self.minimum_pressure_constraint.activate()
        self.pressure_equality_constraints.deactivate()

    def use_equal_pressure_constraint(self):
        """Deactivate the mixer pressure = minimum inlet pressure constraint
        and activate the mixer pressure and all inlet pressures are equal
        constraints. This should only be used when momentum_mixing_type ==
        MomentumMixingType.minimize_and_equality.
        """
        if self.config.momentum_mixing_type != MomentumMixingType.minimize_and_equality:
            _log.warning(
                """use_equal_pressure_constraint() can only be used when
                momentum_mixing_type ==
                MomentumMixingType.minimize_and_equality"""
            )
            return
        self.minimum_pressure_constraint.deactivate()
        self.pressure_equality_constraints.activate()

    def fix_initialization_states(self):
        """
        Iterate over inlet ports and fix all variables.

        For Mixers with pressure equality, we will assume that pressure has been
        correctly specified and not fix pressures.

        Returns:
            None
        """
        inlet_list = self.create_inlet_list()
        for p in inlet_list:
            p_obj = getattr(self, p)
            # Iterate over vars
            for v in p_obj.iter_vars():
                if (
                    self.config.momentum_mixing_type == MomentumMixingType.equality
                    or self.config.momentum_mixing_type
                    == MomentumMixingType.minimize_and_equality
                ) and v.local_name == "pressure":
                    # Don't fix pressure in cases where pressure equality is specified
                    continue
                else:
                    v.fix()

    def initialize_build(
        blk, outlvl=idaeslog.NOTSET, optarg=None, solver=None, hold_state=False
    ):
        """
        Initialization routine for mixer.

        Keyword Arguments:
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None, use
                     default solver options)
            solver : str indicating which solver to use during
                     initialization (default = None, use default solver)
            hold_state : flag indicating whether the initialization routine
                     should unfix any state variables fixed during
                     initialization, **default** - False. **Valid values:**
                     **True** - states variables are not unfixed, and a dict of
                     returned containing flags for which states were fixed
                     during initialization, **False** - state variables are
                     unfixed after initialization by calling the release_state
                     method.

        Returns:
            If hold_states is True, returns a dict containing flags for which
            states were fixed during initialization.
        """
        raise NotImplementedError("Old style initialization is not implemented for the translating mixer")

    def release_state(blk, flags, outlvl=idaeslog.NOTSET):
        """
        Method to release state variables fixed during initialization.

        Keyword Arguments:
            flags : dict containing information of which state variables
                    were fixed during initialization, and should now be
                    unfixed. This dict is returned by initialize if
                    hold_state = True.
            outlvl : sets output level of logging

        Returns:
            None
        """
        inlet_list = blk.create_inlet_list()
        for i in inlet_list:
            i_block = getattr(blk, i + "_state")
            i_block.release_state(flags[i], outlvl=outlvl)

    def _get_stream_table_contents(self, time_point=0):
        io_dict = {}
        inlet_list = self.create_inlet_list()
        for i in inlet_list:
            io_dict[i] = getattr(self, i + "_state")
        if self.config.mixed_state_block is None:
            io_dict["Outlet"] = self.mixed_state
        else:
            io_dict["Outlet"] = self.config.mixed_state_block
        return create_stream_table_dataframe(io_dict, time_point=time_point)

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
        mb_type = self.config.material_balance_type
        if mb_type == MaterialBalanceType.useDefault:
            t_ref = self.flowsheet().time.first()
            mb_type = self.mixed_state[t_ref].default_material_balance_type()

        if hasattr(self, "pressure_equality_constraints"):
            for (t, i), c in self.pressure_equality_constraints.items():
                s = iscale.get_scaling_factor(
                    self.mixed_state[t].pressure, default=1, warning=True
                )
                iscale.constraint_scaling_transform(c, s)

        if hasattr(self, "minimum_pressure"):
            for (t, i), v in self.minimum_pressure.items():
                s = iscale.get_scaling_factor(
                    self.mixed_state[t].pressure, default=1, warning=True
                )
                iscale.set_scaling_factor(v, s)

        if hasattr(self, "minimum_pressure_constraint"):
            for (t, i), c in self.minimum_pressure_constraint.items():
                s = iscale.get_scaling_factor(
                    self.mixed_state[t].pressure, default=1, warning=True
                )
                iscale.constraint_scaling_transform(c, s)

        if hasattr(self, "mixture_pressure"):
            for t, c in self.mixture_pressure.items():
                s = iscale.get_scaling_factor(
                    self.mixed_state[t].pressure, default=1, warning=True
                )
                iscale.constraint_scaling_transform(c, s)

        if hasattr(self, "material_mixing_equations"):
            if mb_type == MaterialBalanceType.componentPhase:
                for (t, p, j), c in self.material_mixing_equations.items():
                    flow_term = self.mixed_state[t].get_material_flow_terms(p, j)
                    s = iscale.get_scaling_factor(flow_term, default=1)
                    iscale.constraint_scaling_transform(c, s, overwrite=False)
            elif mb_type == MaterialBalanceType.componentTotal:
                for (t, j), c in self.material_mixing_equations.items():
                    for i, p in enumerate(self.mixed_state.phase_list):
                        try:
                            ft = self.mixed_state[t].get_material_flow_terms(p, j)
                        except (KeyError, AttributeError):
                            continue  # component not in phase
                        if i == 0:
                            s = iscale.get_scaling_factor(ft, default=1)
                        else:
                            _s = iscale.get_scaling_factor(ft, default=1)
                            s = _s if _s < s else s
                    iscale.constraint_scaling_transform(c, s, overwrite=False)
            elif mb_type == MaterialBalanceType.total:
                pc_set = self.mixed_state.phase_component_set
                for t, c in self.material_mixing_equations.items():
                    for i, (p, j) in enumerate(pc_set):
                        ft = self.mixed_state[t].get_material_flow_terms(p, j)
                        if i == 0:
                            s = iscale.get_scaling_factor(ft, default=1)
                        else:
                            _s = iscale.get_scaling_factor(ft, default=1)
                            s = _s if _s < s else s
                    iscale.constraint_scaling_transform(c, s, overwrite=False)

        if hasattr(self, "enthalpy_mixing_equations"):

            def scale_gen(t):
                for p in self.mixed_state[t].phase_list:
                    yield self.mixed_state[t].get_enthalpy_flow_terms(p)

            for t, c in self.enthalpy_mixing_equations.items():
                s = iscale.min_scaling_factor(scale_gen(t), default=1)
                iscale.constraint_scaling_transform(c, s, overwrite=False)
