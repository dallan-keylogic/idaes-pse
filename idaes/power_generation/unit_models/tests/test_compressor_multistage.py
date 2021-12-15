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

import pytest
from pytest import approx
import pyomo.environ as pyo
from pyomo.util.check_units import (assert_units_consistent, 
                                    assert_units_equivalent)

from idaes.core import FlowsheetBlock
from idaes.generic_models.properties.core.generic.generic_property import (
    GenericParameterBlock)
from idaes.power_generation.unit_models.compressor_multistage import (
    CompressorMultistage)
from idaes.power_generation.properties.natural_gas_PR import get_prop
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import get_solver
import idaes.logger as idaeslog
__author__ = "Douglas Allan"

# Set up solver
solver = get_solver()

@pytest.fixture()
def model():
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    air_species = {"CO2","Ar","H2O","O2","N2"}
    m.fs.props_gas = GenericParameterBlock(
            default=get_prop(air_species, ["Vap"]))
    air_comp = {
        "O2":0.2074,
        "H2O":0.0099,
        "CO2":0.0003,
        "N2":0.7732,
        "Ar":0.0092}
    # air_comp = {
    #     "O2":0.20,
    #     "H2O":0.2,
    #     "CO2":0.2,
    #     "N2":0.2,
    #     "Ar":0.2}
    m.fs.compressor_train = comp = CompressorMultistage(
        default={"property_package":m.fs.props_gas,
                 "num_stages":2,
                 "include_final_cooler":False})
    for key,val in air_comp.items():
        comp.inlet.mole_frac_comp[0,key].fix(val)
    comp.inlet.temperature[0].fix(300)
    comp.inlet.pressure[0].fix(1E5)
    comp.inlet.flow_mol[0].fix(1000)
    
    comp.ratioP[0].fix(1.5)
    comp.cold_gas_temperature[0].fix(273.15+40)
    comp.efficiency_isentropic[0].fix(0.85)
    
    return m

@pytest.mark.component
def test_multistage_compressor(model):
    m = model
    air_species = {"CO2","Ar","H2O","O2","N2"}
    comp = m.fs.compressor_train
    
    assert(degrees_of_freedom(m) == 0)
    m.fs.compressor_train.initialize(outlvl=idaeslog.INFO)
    
    # Ensure correct variables are still fixed
    assert comp.inlet.temperature[0].fixed
    assert comp.inlet.pressure[0].fixed
    assert comp.inlet.flow_mol[0].fixed
    for key in air_species:
        assert comp.inlet.mole_frac_comp[0,key].fixed
    assert comp.ratioP[0].fixed
    assert comp.inlet.pressure[0].fixed
    assert comp.efficiency_isentropic[0].fixed
    
    # Ensure all constraints are active
    # and no temporary constraints survived
    for con in comp.component_data_objects(pyo.Constraint):
        assert con.local_name != "tmp_init_constraint"
        assert con.active
        
    assert(degrees_of_freedom(m) == 0)
    
    # Assert variables have correct values
    assert(comp.compressors[1].outlet.pressure[0].value
           /comp.compressors[1].inlet.pressure[0].value
           == approx(1.5))
    assert (comp.coolers[1].inlet.pressure[0].value 
            == approx(comp.coolers[1].outlet.pressure[0].value))
    assert(comp.outlet.pressure[0].value/comp.inlet.pressure[0].value
           == approx(1.5**2))
    assert(comp.coolers[1].outlet.temperature[0].value
           == approx(273.15+40))
    assert(comp.outlet.temperature[0].value
           == approx(358.4436))
    assert(comp.outlet.flow_mol[0].value 
           == approx(comp.inlet.flow_mol[0].value))
    for key in air_species:
        assert(comp.outlet.mole_frac_comp[0,key].value
               == approx(comp.inlet.mole_frac_comp[0,key].value))

@pytest.mark.component
def test_units(model):
    assert_units_consistent(model)
    assert_units_equivalent(model.fs.compressor_train.work_mechanical[0], 
                            pyo.units.W)
    assert_units_equivalent(model.fs.compressor_train.heat_duty[0], 
                            pyo.units.W)
    assert_units_equivalent(model.fs.compressor_train.ratioP[0], 
                            pyo.units.dimensionless)
    assert_units_equivalent(model.fs.compressor_train.cold_gas_temperature[0], 
                            pyo.units.K)
    
    
# TODO:
# Basic unit testing to ensure right components have and haven't been created
# Test scaling
# Test with IAPWS95 steam being compressed

# if __name__ == "__main__":
#     test_multistage_compressor()