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
Multistage steam turbine for power generation.

Liese, (2014). "Modeling of a Steam Turbine Including Partial Arc Admission
    for Use in a Process Simulation Software Environment." Journal of Engineering
    for Gas Turbines and Power. v136, November
"""
import copy

import pyomo.environ as pyo
from pyomo.network import Arc, Port
from pyomo.common.config import ConfigBlock, ConfigValue, ConfigList, In

from idaes.core import declare_process_block_class, UnitModelBlockData, useDefault
from idaes.generic_models.unit_models import (
    Compressor,
    Heater,
    Separator,
    Mixer,
    MomentumMixingType
)
from idaes.generic_models.unit_models.pressure_changer import ThermodynamicAssumption

from idaes.power_generation.unit_models.helm import HelmValve as SteamValve

from idaes.core.util.initialization import (propagate_state,
                                            fix_state_vars, revert_state_vars )
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util import from_json, to_json, StoreSpec
from idaes.core.util import get_solver
import idaes.core.util.scaling as iscale

import idaes.logger as idaeslog

__author__ = "Douglas Allan"

_log = idaeslog.getLogger(__name__)


def _define_compressor_multistage_config(config):
    config.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag",
            doc="Only False, in a dynamic flowsheet this is psuedo-steady-state.",
        ),
    )
    config.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag",
            doc="Only False, in a dynamic flowsheet this is psuedo-steady-state.",
        ),
    )
    config.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package of gas being compressed",
            doc="""Property parameter object used to define property calculations,
                **default** - useDefault.
                **Valid values:** {
                **useDefault** - use default package from parent model or flowsheet,
                **PropertyParameterObject** - a PropertyParameterBlock object.}""",
        ),
    )
    config.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
                and used when constructing these,
                **default** - None.
                **Valid values:** {
                see property package for documentation.}""",
        ),
    )
    config.declare(
        "num_stages",
        ConfigValue(
            default=2,
            domain=int,
            description="Number of compressors used. Default=2",
        ),
    )
    config.declare(
        "include_final_cooler",
        ConfigValue(
            domain=In(pyo.Boolean),
            default=True,
            description="Whether a heat exchanger should be included after "
                        "the final compressor stage. Default=True",
        ),
    )
    config.declare(
        "equal_ratioP",
        ConfigValue(
            domain=In(pyo.Boolean),
            default=True,
            description="Whether all compressors should be constrained to have "
                        "the same ratioP. Default=True",
        ),
    )
    config.declare(
        "equal_efficiency_isentropic",
        ConfigValue(
            domain=In(pyo.Boolean),
            default=True,
            description="Whether all compressors should be constrained to have "
                        "the same isentropic efficiency. Default=True",
        ),
    )
    config.declare(
        "equal_cold_gas_temperature",
        ConfigValue(
            domain=In(pyo.Boolean),
            default=True,
            description="Whether all coolers should be constrained to have "
                        "the same outlet temperature for gas. Default=True",
        ),
    )


@declare_process_block_class(
    "CompressorMultistage",
    doc="Multistage compressor with intercooling",
)
class CompressorMultistageData(UnitModelBlockData):
    CONFIG = ConfigBlock()
    _define_compressor_multistage_config(CONFIG)

    def build(self):
        super().build()
        config = self.config
        compressor_cfg = {
            "dynamic": config.dynamic,
            "has_holdup": config.has_holdup,
            "property_package": config.property_package,
            "property_package_args": config.property_package_args,
            "thermodynamic_assumption": ThermodynamicAssumption.isentropic
        }
        cooler_cfg = {
            "dynamic": config.dynamic,
            "has_holdup": config.has_holdup,
            "property_package": config.property_package,
            "property_package_args": config.property_package_args
        }

        n_comp = self.config.num_stages
        n_coolers = n_comp - 1 + config.include_final_cooler # Bool casts to int
        n_comp_idx = pyo.RangeSet(n_comp)
        n_comp_m1_idx = pyo.RangeSet(n_comp-1)
        n_coolers_idx = pyo.RangeSet(n_coolers)
   
        self.compressors = Compressor(n_comp_idx, default=compressor_cfg)
        self.coolers = Heater(n_coolers_idx, default=cooler_cfg)
        
        t0 = self.flowsheet().time.first()
        
        
        # Clever code is the path to the dark side
        # def lift_var(name,component,doc_string):
        #     equal_value = getattr(config,"equal_"+name)
        #     if equal_value:
        #         low_level_var = getattr(component[1],name)
        #         init = low_level_var[t0].value
        #         bounds = low_level_var[t0].bounds
        #         setattr(self,name,pyo.Var(self.flowsheet().time,
        #             initialize=init,bounds = bounds, doc = doc_string)
                
        #         s
        #         @self.Constraint(self.flowsheet().time, n_comp_idx)
        #         def ratioP_eqn(b,t,i):
        #             return b.compressors[i].ratioP[t] == b.ratioP[t]
        #     else:
                
        # For ratioP, efficiency_isentropic, and cold_gas_temperature, we
        # either create a single higher level variable and constrain the lower
        # level component variables to equal that variable, or create a higher
        # level indexed reference to the lower level component variables,
        # depending on how the user configured the compound component
        if config.equal_ratioP:
            self.ratioP = pyo.Var(self.flowsheet().time,
                initialize=self.compressors[1].ratioP[t0].value,
                bounds = self.compressors[1].ratioP[t0].bounds,
                doc = "Pressure ratio for all compressors in multistage series.")
            
            @self.Constraint(self.flowsheet().time, n_comp_idx)
            def ratioP_eqn(b,t,i):
                return b.compressors[i].ratioP[t] == b.ratioP[t]
        else:
            self.ratioP = pyo.Reference(self.compressors[:].ratioP[:].value)
        
        if config.equal_efficiency_isentropic:
            self.efficiency_isentropic = pyo.Var(
                self.flowsheet().time,
                initialize=self.compressors[1].efficiency_isentropic[t0].value,
                bounds = self.compressors[1].efficiency_isentropic[t0].bounds,
                doc = "Isentropic efficiency for all compressors "
                                        "in multistage series.")
            
            @self.Constraint(self.flowsheet().time, n_comp_idx)
            def efficiency_isentropic_eqn(b,t,i):
                return (b.compressors[i].efficiency_isentropic[t] 
                        == b.efficiency_isentropic[t])
        else:
            self.efficiency_isentropic = pyo.Reference(
                self.compressors[:].efficiency_isentropic[:])
        
        if config.equal_cold_gas_temperature:
            s1_metadata = config.property_package.get_metadata()
            temp_units = s1_metadata.get_derived_units("temperature")
                
            self.cold_gas_temperature = pyo.Var(
                self.flowsheet().time,
                initialize = 1,
                #bounds = self.heat_exchangers[1].hot_side\
                #    .properties_out[t0].temperature.bounds,
                units = temp_units,
                doc = "Temperature reached before next compression stage")
            
            @self.Constraint(self.flowsheet().time, n_coolers_idx)
            def cold_gas_temperature_eqn(b,t,i):
                return (b.coolers[i].outlet.temperature[t]
                        == b.cold_gas_temperature[t])
        else:
            self.cold_gas_temperature = pyo.Reference(
                self.coolers[:].outlet.temperature[:])
        
        @self.Expression(self.flowsheet().time)
        def work_mechanical(b,t):
            return sum(b.compressors[i].work_mechanical[t] for i in n_comp_idx)
        
        @self.Expression(self.flowsheet().time)
        def heat_duty(b,t):
            return sum(b.coolers[i].heat_duty[t] for i in n_coolers_idx)
        
        def cold_gas_rule(b,i):
            return {
                "source": b.coolers[i].outlet,
                "destination": b.compressors[i+1].inlet,
            }
        
        def hot_gas_rule(b,i):
            return {
                "source": b.compressors[i].outlet,
                "destination": b.coolers[i].inlet,
            }
        
        self.cold_gas = Arc(n_comp_m1_idx, rule=cold_gas_rule)
        self.hot_gas = Arc(n_coolers_idx, rule=hot_gas_rule)
        
        self.inlet = Port(extends=self.compressors[1].inlet)
        if config.include_final_cooler:
            self.outlet = Port(
                extends=self.coolers[n_coolers].outlet)
        else:
            self.outlet = Port(
                extends=self.compressors[n_comp].outlet)
        
        pyo.TransformationFactory("network.expand_arcs").apply_to(self)

    def initialize(
        self,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None
    ):
        """
        Initialize

        Args:
            outlvl: logging level default is NOTSET, which inherits from the
                parent logger
            solver: the NL solver
            optarg: solver arguments, default is None
            
        Returns:
            None
        """
        # Setup loggers
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")
        opt = get_solver(solver, optarg)
        # Store initial model specs, restored at the end of initializtion, so
        # the problem is not altered.  This can restore fixed/free vars,
        # active/inactive constraints, and fixed variable values.
        # sp = StoreSpec.value_isfixed_isactive(only_fixed=True)
        # istate = to_json(self, return_dict=True, wts=sp)

        n_comp = self.config.num_stages
        n_coolers = (self.config.num_stages - 1 
                         + self.config.include_final_cooler)
        
        
        # Iterate over unit level variables and store their fixedness status
        component_level_vars = [vbl for vbl in self.inlet.vars.values()]
        component_level_vars += [self.ratioP, self.efficiency_isentropic, 
                                      self.cold_gas_temperature]
        
        component_level_vars += [vbl for vbl in self.outlet.vars.values()]
            
        component_level_constraints = []
        if self.config.equal_ratioP:
            component_level_constraints.append(self.ratioP_eqn)
        if self.config.equal_efficiency_isentropic:
            component_level_constraints.append(self.efficiency_isentropic_eqn)
        if self.config.equal_cold_gas_temperature:
            component_level_constraints.append(self.cold_gas_temperature_eqn)
        #TODO support user component level variables and constraints
        
        variable_fixed = {}
        for vbl in component_level_vars:
            variable_fixed[vbl] = {}
            for idx, val in vbl.items():
                variable_fixed[vbl][idx] = val.fixed
            
        constraint_active = {}
        for con in component_level_constraints:
            constraint_active[con] = {}
            for idx, val in con.items():
                constraint_active[con][idx] = val.active
            con.deactivate()
        
        
        t0 = self.flowsheet().time.first()
        
        prop_in = self.compressors[1].control_volume.properties_in
        
        if self.config.include_final_cooler:
            prop_out = self.coolers[n_comp].control_volume.properties_out
        else:
            prop_out = self.compressors[n_comp].control_volume.properties_out
        
        # This initialization assumes that process variables should be fixed for
        # all times in flowsheet.time. It's unclear if that's something we should
        # require or not, but technically dynamics aren't supported with this
        # unit model anyway. 
        #TODO: revisit later---better dynamic behavior and init props only once
        if all(val.fixed for val in self.ratioP.values()):
            ratioP = {idx:val.value for idx,val in self.ratioP.items()}
        elif self.config.equal_ratioP:
            # Attempt to infer desired overall pressure change of compressor
            # train from pressure of final block
            
            # Create Pyomo expression before initializing to ensure 
            # pressure is generated before blocks are solved
            overall_ratioP = {t:prop_out[t].pressure/prop_in[t].pressure
                               for t in self.flowsheet().time}
            prop_in.initialize()
            prop_out.initialize()
            ratioP = {t:pyo.value(overall_ratioP[t]**(1/n_comp))
                      for t in self.flowsheet().time}
        else:
            raise RuntimeError(f"In initialization of block {self.name}, "
                               "equal_ratioP is false, but ratioP is not fixed "
                               "for all times and compressors. Unclear how "
                               "to initialize block.")
        if all(val.fixed for val in self.efficiency_isentropic.values()):
            efficiency_isentropic = {
                idx:val.value for idx,val in self.ratioP.items()}
        else:
            raise RuntimeError(f"In initialization of block {self.name}, "
                               "efficiency_isentropic is not fixed for all "
                               "times and compressors. Unclear how to "
                               "initialize block.")
        if all(val.fixed for val in self.cold_gas_temperature.values()):
            cold_gas_temp = {idx:val.value 
                             for idx,val in self.cold_gas_temperature.items()}
        elif self.config.equal_cold_gas_temperature:
            if self.config.include_final_cooler:
                # Try to infer cold_gas_temp from temperature of output cooler
                cold_gas_temp_expr = {t:prop_out[t].temperature
                                      for t in self.flowsheet().time}
                prop_out.initialize()
            else:
                # Assume compressor train is isothermal
                cold_gas_temp_expr = {t:prop_in[t].temperature
                                      for t in self.flowsheet().time}
                prop_in.initialize()
            cold_gas_temp = {t:pyo.value(cold_gas_temp_expr[t])}
        else: 
            raise RuntimeError(f"In initialization of block {self.name}, "
                               "equal_cold_gas_temperature is false, but "
                               "cold_gas_temperature is not fixed "
                               "for all times and coolers. Unclear how "
                               "to initialize block.")
            
        
        self.compressors.deactivate()
        self.hot_gas.deactivate()
        self.coolers.deactivate()
        self.cold_gas.deactivate()
        
        # for t in self.flowsheet().time:
        #     self.ratioP[t].fix()
        #     self.efficiency_isentropic[t].fix()
        #     self.cold_gas_temperature[t].fix()

        
        # self.ratioP_eqn.deactivate()
        # self.efficiency_isentropic_eqn.deactivate()
        # self.cold_gas_temperature_eqn.deactivate()
            
        for i in range(1,n_comp+1):
            # Initialize compressor
            self.compressors[i].activate()
            # self.ratioP_eqn[i].activate()
            # self.efficiency_isentropic_eqn[i].activate()
            for t in self.flowsheet().time:
                if self.config.equal_ratioP:
                    self.compressors[i].ratioP[t].fix(ratioP[t])
                else:
                    self.compressors[i].ratioP[t].fix(ratioP[(i,t)])
                if self.config.equal_efficiency_isentropic:
                    self.compressors[i].efficiency_isentropic[t].fix(
                            efficiency_isentropic[t])
                else:
                    raise NotImplementedError("blag")
            self.compressors[i].initialize(outlvl=outlvl, optarg=optarg, solver=solver)
            # with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            #     res = opt.solve(self, tee=slc.tee)
            
            for t in self.flowsheet().time:
                if self.config.equal_ratioP:
                    self.compressors[i].ratioP[t].unfix()
                else:
                    raise NotImplementedError("blag")
                if self.config.equal_efficiency_isentropic:
                    self.compressors[i].efficiency_isentropic[t].unfix()
                else:
                    raise NotImplementedError("blag")
            self.compressors[i].deactivate()
            # self.ratioP_eqn[i].deactivate()
            # self.efficiency_isentropic_eqn[i].deactivate()
            
            # Initialize heat exchanger
            if i != n_comp or self.config.include_final_cooler:
                propagate_state(arc=self.hot_gas[i], direction="forward")
                self.coolers[i].activate()
                
                if self.config.equal_cold_gas_temperature:
                    def tmp_rule(b, t):
                        return (b.control_volume.properties_out[t].temperature
                                == cold_gas_temp[t])
                    self.coolers[i].tmp_init_constraint = pyo.Constraint(
                        self.flowsheet().time, rule=tmp_rule)
                else:
                    raise NotImplementedError("blag")

                       
               
                self.coolers[i].initialize(outlvl=outlvl,
                                           optarg=optarg, solver=solver)
                
                # Need to perform a unit level solve in order to take the
                # temperature constraint into account
                # TODO: Is it better to write a temporary constraint to the
                # heat exchanger?
                # flags = fix_state_vars(self.coolers[i].control_volume.properties_in)
                # with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                #     res = opt.solve(self, tee=slc.tee)
                # revert_state_vars(self.coolers[i].control_volume.properties_in,
                #                    flags)
                self.coolers[i].del_component(self.coolers[i].tmp_init_constraint)
                self.coolers[i].deactivate()
                
            if i!=n_comp:
                propagate_state(arc=self.cold_gas[i], direction="forward")
        self.compressors.activate()
        self.hot_gas.activate()
        self.coolers.activate()
        self.cold_gas.activate()

        # with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
        #     res = opt.solve(self, tee=slc.tee)
        # init_log.info_high("Initialization Step 3 {}."
        #                    .format(idaeslog.condition(res)))
        for vbl in component_level_vars:
            for idx, val in vbl.items():
                if not variable_fixed[vbl][idx]:
                    vbl[idx].unfix()
        for con in component_level_constraints:
            for idx, val in con.items():
                if constraint_active[con][idx]:
                    con[idx].activate()
        



    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
        #t0 = self.flowsheet().time.first()
        
        #n_compressors = self.config.num_stages
        n_coolers = (self.config.num_stages - 1 
                          + self.config.include_final_cooler)
        
        for t in self.flowsheet().time:
            for i in range(1,n_coolers+1):
                sf_T = iscale.get_scaling_factor(\
                    self.coolers[i].outlet.temperature[t], 
                    default=300, warning=True)
                iscale.constraint_scaling_transform(
                    self.cold_gas_temperature_eqn[t,i], sf_T, overwrite=False)
            if iscale.get_scaling_factor(self.cold_gas_temperature) is None:
                iscale.set_scaling_factor(self.cold_gas_temperature, sf_T)
            
