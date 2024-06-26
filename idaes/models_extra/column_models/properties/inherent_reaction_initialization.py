from pyomo.environ import (
    Block,
    check_optimal_termination,
    ConcreteModel,
    Constraint,
    log,
    Set,
    Param,
    value,
    Var,
    units as pyunits,
)
from pyomo.common.config import ConfigBlock, ConfigDict, ConfigValue, In, Bool
from pyomo.util.calc_var_value import calculate_variable_from_constraint

from idaes.core.base.components import __all_components__
from idaes.core.base.phases import (
    __all_phases__,
)
from idaes.core.util.initialization import (
    solve_indexed_blocks,
)
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_activated_constraints,
)
from idaes.core.util.exceptions import (
    BurntToast,
    InitializationError,
)
from idaes.core.solvers import get_solver
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
from idaes.core.initialization.initializer_base import InitializerBase

from idaes.models.properties.modular_properties.base.utility import (
    StateIndex,
)
from idaes.models.properties.modular_properties.phase_equil.henry import HenryType
from idaes.models.properties.modular_properties.base.generic_property import _initialize_critical_props
# Set up logger
_log = idaeslog.getLogger(__name__)

def _safe_set_value(var, val):
    if var.lb is not None and val < var.lb:
        var.set_value(var.lb)
    elif var.ub is not None and val > var.ub:
        var.set_value(var.ub)
    else:
        var.set_value(val)

class ModularPropertiesInherentReactionsInitializer(InitializerBase):
    """
    Modified form of general Initializer for modular property packages
    for use with inherent reactions.

    This Initializer uses a hierarchical routine to initialize the
    property package using the following steps:

    TODO add more details

    1. Initialize bubble and dew point calculations (if present)
    2. Estimate vapor-liquid equilibrium T_eq (if present)
    3. Solve for phase-equilibrium conditions
    4. Initialize all remaining properties

    The Pyomo solve_strongly_connected_components method is used at each
    step to converge the problem.

    Note that for systems without vapor-liquid equilibrium the generic
    BlockTriangularizationInitializer is probably sufficient for initializing
    the property package.

    """

    CONFIG = InitializerBase.CONFIG()
    CONFIG.declare(
        "solver",
        ConfigValue(
            default=None,
            description="Solver to use for initialization",
        ),
    )
    CONFIG.declare(
        "solver_options",
        ConfigDict(
            implicit=True,
            description="Dict of options to pass to solver",
        ),
    )
    CONFIG.declare(
        "solver_writer_config",
        ConfigDict(
            implicit=True,
            description="Dict of writer_config arguments to pass to solver",
        ),
    )
    CONFIG.declare(
        "calculate_variable_options",
        ConfigDict(
            implicit=True,
            description="Dict of options to pass to 1x1 block solver",
            doc="Dict of options to pass to calc_var_kwds argument in "
            "solve_strongly_connected_components method. NOTE: models "
            "involving ExternalFunctions must set "
            "'diff_mode=differentiate.Modes.reverse_numeric'",
        ),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._solver = None

    def initialization_routine(
        self,
        model: Block,
    ):
        """
        Sequential initialization routine for modular properties.

        Args:
            model: model to be initialized

        Returns:
            None
        """
        # Setup loggers
        init_log = idaeslog.getInitLogger(
            model.name, self.config.output_level, tag="properties"
        )
        solve_log = idaeslog.getSolveLogger(
            model.name, self.config.output_level, tag="properties"
        )

        # Create solver object
        solver_obj = get_solver(
            solver=self.config.solver,
            solver_options=self.config.solver_options,
            writer_config=self.config.solver_writer_config,
        )

        init_log.info("Starting initialization routine")

        index_data = model.index_set().data()
        idx0 = index_data[0]
        params = model[idx0].params

        for k in model.values():
            # Deactivate the constraints specific for outlet block i.e.
            # when defined state is False
            if k.config.defined_state is False:
                try:
                    k.sum_mole_frac_out.deactivate()
                except AttributeError:
                    pass

                if hasattr(k, "inherent_equilibrium_constraint") and (
                    not k.params._electrolyte
                    or k.params.config.state_components == StateIndex.true
                ):
                    k.inherent_equilibrium_constraint.deactivate()

        # ---------------------------------------------------------------------
        # If present, initialize bubble, dew, and critical point calculations
        for k in model.values():
            T_units = k.params.get_metadata().default_units.TEMPERATURE

            # List of bubble and dew point constraints
            cons_list = [
                "eq_pressure_dew",
                "eq_pressure_bubble",
                "eq_temperature_dew",
                "eq_temperature_bubble",
                "eq_mole_frac_tbub",
                "eq_mole_frac_tdew",
                "eq_mole_frac_pbub",
                "eq_mole_frac_pdew",
                "log_mole_frac_tbub_eqn",
                "log_mole_frac_tdew_eqn",
                "log_mole_frac_pbub_eqn",
                "log_mole_frac_pdew_eqn",
                "mole_frac_comp_eq",
                "log_mole_frac_comp_eqn",
            ]

            # Critical point
            with k.lock_attribute_creation_context():
                # Only need to look for one, as it is all-or-nothing
                if hasattr(k, "pressure_crit"):
                    # Initialize critical point properties
                    _initialize_critical_props(k)
                    # Add critical point constraints to cons_list
                    ref_phase = k._get_critical_ref_phase()
                    p_config = k.params.get_phase(ref_phase).config
                    cons_list += (
                        p_config.equation_of_state.list_critical_property_constraint_names()
                    )

            # Bubble temperature initialization
            if hasattr(k, "_mole_frac_tbub"):
                model._init_Tbub(k, T_units)

            # Dew temperature initialization
            if hasattr(k, "_mole_frac_tdew"):
                model._init_Tdew(k, T_units)

            # Bubble pressure initialization
            if hasattr(k, "_mole_frac_pbub"):
                model._init_Pbub(k, T_units)

            # Dew pressure initialization
            if hasattr(k, "_mole_frac_pdew"):
                model._init_Pdew(k, T_units)

            # Solve bubble, dew, and critical point constraints
            for c in k.component_objects(Constraint):
                # Deactivate all constraints not associated with bubble and dew
                # points or critical points
                if c.local_name not in cons_list:
                    c.deactivate()

        # If StateBlock has active constraints (i.e. has bubble, dew, or critical
        # point calculations), solve the block to converge these
        for b in model.values():
            if number_activated_constraints(b) > 0:
                if not degrees_of_freedom(b) == 0:
                    raise InitializationError(
                        f"{b.name} Unexpected degrees of freedom during "
                        f"initialization at bubble, dew, and critical point step: "
                        f"{degrees_of_freedom(b)}."
                    )
        if number_activated_constraints(model) > 0:
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                solve_indexed_blocks(solver_obj, model, tee=slc.tee)
        init_log.info("Bubble, dew, and critical point initialization completed.")

        # ---------------------------------------------------------------------
        # Calculate _teq if required
        for k in model.values():
            if k.params.config.phases_in_equilibrium is not None and (
                not k.config.defined_state or k.always_flash
            ):
                for pp in k.params._pe_pairs:
                    k.params.config.phase_equilibrium_state[pp].calculate_teq(k, pp)

        init_log.info("Equilibrium temperature initialization completed.")

        # ---------------------------------------------------------------------
        # Initialize flow rates and compositions
        for k in model.values():
            k.params.config.state_definition.state_initialization(k)

        if params.config.state_components == StateIndex.apparent:
            _initialize_inherent_reactions(model)

        for k in model.values():
            if k.params._electrolyte:
                # TODO add initialization of reactions here too
                if k.params.config.state_components == StateIndex.true:
                    # First calculate initial values for apparent species flows
                    for p, j in k.params.apparent_phase_component_set:
                        calculate_variable_from_constraint(
                            k.flow_mol_phase_comp_apparent[p, j],
                            k.true_to_appr_species[p, j],
                        )
                    # Need to calculate all flows before doing mole fractions
                    for p, j in k.params.apparent_phase_component_set:
                        sum_flow = sum(
                            k.flow_mol_phase_comp_apparent[p, jj]
                            for jj in k.params.apparent_species_set
                            if (p, jj) in k.params.apparent_phase_component_set
                        )
                        if value(sum_flow) == 0:
                            x = 1
                        else:
                            x = value(k.flow_mol_phase_comp_apparent[p, j] / sum_flow)
                        lb = k.mole_frac_phase_comp_apparent[p, j].lb
                        if lb is not None and x <= lb:
                            k.mole_frac_phase_comp_apparent[p, j].set_value(lb)
                        else:
                            k.mole_frac_phase_comp_apparent[p, j].set_value(x)
                elif k.params.config.state_components == StateIndex.apparent:
                    # Was handled elsewhere
                    pass

                else:
                    raise BurntToast(
                        "Electrolyte basis other than true or apparent chosen. "
                        "This should not be possible, please contact the IDAES developers."
                    )

            # If state block has phase equilibrium, use the average of all
            # _teq's as an initial guess for T
            if (
                k.params.config.phases_in_equilibrium is not None
                and isinstance(k.temperature, Var)
                and not k.temperature.fixed
            ):
                k.temperature.value = value(
                    sum(k._teq[i] for i in k.params._pe_pairs) / len(k.params._pe_pairs)
                )

        init_log.info("State variable initialization completed.")

        # ---------------------------------------------------------------------
        Tfix = {}  # In enth based state defs, need to also fix T until later
        for k, b in model.items():
            if b.params.config.phase_equilibrium_state is not None and (
                not b.config.defined_state or b.always_flash
            ):
                if not b.temperature.fixed:
                    b.temperature.fix()
                    Tfix[k] = True
                for c in b.component_objects(Constraint):
                    # Activate common constraints
                    if c.local_name in (
                        "total_flow_balance",
                        "component_flow_balances",
                        "sum_mole_frac",
                        "phase_fraction_constraint",
                        "mole_frac_phase_comp_eq",
                        "mole_frac_comp_eq",
                    ):
                        c.activate()
                    if c.local_name == "log_mole_frac_phase_comp_eqn":
                        c.activate()
                        for p, j in b.params._phase_component_set:
                            calculate_variable_from_constraint(
                                b.log_mole_frac_phase_comp[p, j],
                                b.log_mole_frac_phase_comp_eqn[p, j],
                            )
                    elif c.local_name == "equilibrium_constraint":
                        # For systems where the state variables fully define the
                        # phase equilibrium, we cannot activate the equilibrium
                        # constraint at this stage.
                        if "flow_mol_phase_comp" not in b.define_state_vars():
                            c.activate()

                for pp in b.params._pe_pairs:
                    # Activate formulation specific constraints
                    b.params.config.phase_equilibrium_state[
                        pp
                    ].phase_equil_initialization(b, pp)

        if number_activated_constraints(model) > 0:
            dof = degrees_of_freedom(model)
            if dof == 0:
                with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                    solve_indexed_blocks(solver_obj, [model], tee=slc.tee)
            elif dof > 0:
                raise InitializationError(
                    f"{model.name} Unexpected degrees of freedom during "
                    f"initialization at phase equilibrium step: {dof}."
                )
            # Skip solve if DoF < 0 - this is probably due to a
            # phase-component flow state with flash

        init_log.info("Phase equilibrium initialization completed.")

        # ---------------------------------------------------------------------
        # Initialize other properties
        for k, b in model.items():
            for c in b.component_objects(Constraint):
                # Activate all constraints except flagged do_not_initialize
                if c.local_name not in (
                    b.params.config.state_definition.do_not_initialize
                ):
                    c.activate()
            if k in Tfix:
                b.temperature.unfix()

            # Initialize log-form variables
            log_form_vars = [
                "act_phase_comp",
                "act_phase_comp_apparent",
                "act_phase_comp_true",
                "conc_mol_phase_comp",
                "conc_mol_phase_comp_apparent",
                "conc_mol_phase_comp_true",
                "mass_frac_phase_comp",
                "mass_frac_phase_comp_apparent",
                "mass_frac_phase_comp_true",
                "molality_phase_comp",
                "molality_phase_comp_apparent",
                "molality_phase_comp_true",
                "mole_frac_comp",  # Might have already been initialized
                "mole_frac_phase_comp",  # Might have already been initialized
                "mole_frac_phase_comp_apparent",
                "mole_frac_phase_comp_true",
                "pressure_phase_comp",
                "pressure_phase_comp_apparent",
                "pressure_phase_comp_true",
            ]

            for prop in log_form_vars:
                if b.is_property_constructed("log_" + prop):
                    comp = getattr(b, prop)
                    lcomp = getattr(b, "log_" + prop)
                    for k2, v in lcomp.items():
                        c = value(comp[k2])
                        if c <= 0:
                            c = 1e-8
                        lc = log(c)
                        v.set_value(value(lc))

        if number_activated_constraints(model) > 0:
            dof = degrees_of_freedom(model)
            if dof == 0:
                with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                    result = solve_indexed_blocks(solver_obj, model, tee=slc.tee)
            elif dof > 0:
                raise InitializationError(
                    f"{model.name} Unexpected degrees of freedom during "
                    f"initialization at phase equilibrium step: {dof}."
                )
		# Skip solve if DoF < 0 - this is probably due to a
		# phase-component flow state with flash

        init_log.info("Property initialization routine finished.")

        return result
    
def _initialize_inherent_reactions(indexed_blk):
    second_iteration=True
    model = ConcreteModel()
    index_data = indexed_blk.index_set().data()
    idx0 = index_data[0]
    params = indexed_blk[idx0].params
    model.idx_set = Set(initialize=index_data)
    model.inherent_reaction_idx = Set(initialize=[j for j in params.inherent_reaction_idx])
    model.true_species_set = Set(initialize=[j for j in params.true_species_set])
    def rule_block_gen(blk, *args):
        idx = args
        blk.reduced_inherent_reaction_extent = Var(
            model.inherent_reaction_idx,
            initialize=0,
            units=pyunits.dimensionless,
            doc="Apparent extent of inherent reactions",
        )
        blk.pseudo_mole_frac_comp_true = Var(
            model.true_species_set,
            initialize=1/len(model.true_species_set),
            units=pyunits.dimensionless,
            bounds=[0, 1.001],
            doc="Moles of true species j divided by total number of moles of apparent species",
        )

        @blk.Expression()
        def reduced_total_mol(b):
            return sum(b.pseudo_mole_frac_comp_true[j] for j in model.true_species_set)
        
        @blk.Param(model.true_species_set)
        def mole_frac_comp_initial(b, j):
            # Don't like accessing objects in initialization functions
            # but I don't see a better way without adding a ton of verbosity
            # This needs to be changed if dissociation species are present
            if j in params.apparent_species_set:
                return value(indexed_blk[idx].mole_frac_comp[j])
            else:
                return 0


        blk.log_equil_const = Param(
            model.inherent_reaction_idx,
            initialize=0,
            mutable=True
        )
        for r in model.inherent_reaction_idx:
            blk.log_equil_const[r] = value(indexed_blk[idx].log_k_eq[r])

        @blk.Constraint(model.true_species_set)
        def material_balance_eqn(b, j):
            return b.pseudo_mole_frac_comp_true[j] == (
                b.mole_frac_comp_initial[j]
                + sum(
                    params.inherent_reaction_stoichiometry[r, "Liq", j]
                    * b.reduced_inherent_reaction_extent[r]
                    for r in model.inherent_reaction_idx
                )
            )

    model.pseudo_state_blocks = Block(
        model.idx_set,
        rule=rule_block_gen
    )
    @model.Objective()
    def obj(m):
        return sum(
            -sum(
                blk.log_equil_const[r]
                * blk.reduced_inherent_reaction_extent[r]
                for r in m.inherent_reaction_idx
            )
            - blk.reduced_total_mol * log(blk.reduced_total_mol)
            + sum(
                blk.pseudo_mole_frac_comp_true[j]
                * log(blk.pseudo_mole_frac_comp_true[j])
                for j in m.true_species_set
            )
            for blk in m.pseudo_state_blocks.values()
        )
    assert degrees_of_freedom(model) == len(model.idx_set) * len(model.inherent_reaction_idx)

    from idaes.core.solvers import get_solver
    solver_obj = get_solver(
        "ipopt",
        options={
            "halt_on_ampl_error": "no",
            "bound_relax_factor": 0,
            # "tiny_step_tol": 1e-10,
            # "jac_c_constant": "yes"
        }
    )
    res = solver_obj.solve(model, tee=False)
    if not check_optimal_termination(res):
        raise InitializationError(
            f"{indexed_blk.name} failed to initialize successfully. Please check "
            f"the output logs for more information."
        )
    def propagate_back_to_indexed_blk():
        for idx in model.idx_set:
            for r in model.inherent_reaction_idx:
                _safe_set_value(
                    indexed_blk[idx].apparent_inherent_reaction_extent[r],
                    value(
                        model.pseudo_state_blocks[idx].reduced_inherent_reaction_extent[r]
                        * indexed_blk[idx].flow_mol
                    )
                )

            for j in model.true_species_set:
                _safe_set_value(
                    indexed_blk[idx].flow_mol_phase_comp_true["Liq", j],
                    value(
                        model.pseudo_state_blocks[idx].pseudo_mole_frac_comp_true[j]
                        * indexed_blk[idx].flow_mol
                    )
                )
                _safe_set_value(
                    indexed_blk[idx].mole_frac_phase_comp_true["Liq", j],
                    value(
                        model.pseudo_state_blocks[idx].pseudo_mole_frac_comp_true[j]
                        / model.pseudo_state_blocks[idx].reduced_total_mol
                    )
                )
    propagate_back_to_indexed_blk()

    if second_iteration:
        for idx, blk in model.pseudo_state_blocks.items():
            for r in model.inherent_reaction_idx:
                blk.log_equil_const[r] = (
                    value(
                        indexed_blk[idx].log_k_eq[r]
                        - sum(
                            params.inherent_reaction_stoichiometry[r, "Liq", j]
                            * indexed_blk[idx].Liq_log_gamma[j]
                            for j in params.true_species_set
                        )
                    )
                )
        res = solver_obj.solve(model, tee=False)
        propagate_back_to_indexed_blk()
