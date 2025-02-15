#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
from idaes.core.util.scaling import set_scaling_factor
from pyomo.environ import (
    ConcreteModel,
    value,
    Constraint,
    Objective,
    Var,
    Expression,
    Set,
    TransformationFactory,
    units as pyunits,
    check_optimal_termination,
    assert_optimal_termination,
    Block,
    Param
)
from pyomo.network import Arc, SequentialDecomposition

import pyomo.environ as pyo
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from idaes.core import FlowsheetBlock
from watertap.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import Feed, Separator, Mixer, Product
from idaes.models.unit_models.translator import Translator
from idaes.models.unit_models.separator import SplittingType
from idaes.models.unit_models.mixer import MomentumMixingType
from idaes.models.unit_models.heat_exchanger import (
    HeatExchanger,
    HeatExchangerFlowPattern,
)
from idaes.core import UnitModelCostingBlock
import idaes.core.util.scaling as iscale
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
from idaes.core.surrogate.surrogate_block import SurrogateBlock

from watertap.unit_models.mvc.components import Evaporator, Compressor, Condenser
from watertap.unit_models.mvc.components.lmtd_chen_callback import (
    delta_temperature_chen_callback,
)
from watertap.unit_models.mvc.tests.test_mvc import initialize
from watertap.unit_models.pressure_changer import Pump
import watertap.property_models.seawater_prop_pack as props_sw
import watertap.property_models.water_prop_pack as props_w
from watertap.costing import WaterTAPCosting
import math
import numpy as np
import pandas as pd

# Import reaktoro-pse and reaktoro
from reaktoro_pse.reaktoro_block import ReaktoroBlock
import reaktoro

# Import scaling objects
from idaes.core.util.scaling import (
    calculate_scaling_factors,
    set_scaling_factor,
    constraint_scaling_transform,
)


def single_run(material='stainless_steel_316',
         do=0):
    # build, set operating conditions, initialize for simulation
    m = build(material=material)
    set_operating_conditions(m)
    add_Q_ext(m, time_point=m.fs.config.time)
    initialize_system(m)
    # rescale costs after initialization because scaling depends on flow rates
    scale_costs(m)
    fix_outlet_pressures(m)  # outlet pressure are initially unfixed for initialization
    m.fs.pretreatment.acid_addition["HCl"].fix(0)
    # print(dir(m.fs.costing))
    # assert False
    # set up for minimizing Q_ext in first solve
    # should be 1 DOF because Q_ext is unfixed
    print("DOF after initialization: ", degrees_of_freedom(m))
    m.fs.objective = Objective(expr=m.fs.Q_ext[0])

    print("\n***---First solve - simulation results---***")
    solver = get_solver()
    results = solve(m, solver=solver, tee=False)
    print("Termination condition: ", results.solver.termination_condition)
    assert_optimal_termination(results)
    display_metrics(m)
    display_design(m)

    print("\n***---Second solve - optimization ---***")
    m.fs.Q_ext[0].fix(0)  # no longer want external heating in evaporator
    del m.fs.objective
    set_up_optimization(m)
    results = solve(m, solver=solver, tee=False)
    print("Termination condition: ", results.solver.termination_condition)
    display_metrics(m)
    display_design(m)

    print("\n***---Third solve - optimization with calcite scaling tendency---***")
    print('DOF before adding reaktoro: ', degrees_of_freedom(m))
    m.fs.pretreatment.acid_addition["HCl"].unfix()
    add_scaling_tendencies(m)
    solve_with_reaktoro(m)
    m.fs.evaporator.eq_reaktoro_properties.reaktoro_model.outputs.display()
    display_reaktoro_metrics(m)

    print("\n***---Fourth solve - optimization with bound on calcite scaling tendency---***")
    setup_optimization_with_reaktoro(m)
    print('DOF for optimization with reaktoro: ', degrees_of_freedom(m))
    solve_with_reaktoro(m)
    display_reaktoro_metrics(m)
    assert False  # This will stop the code here

    # print("\n***---Second solve - optimization with corrosion rate surrogate---***")
    # add_evap_hx_material_factor_equal_constraint(m)
    # add_corrosion_rate_surrogate(m)
    # set_surrogate_conditions(m,do)
    # m.fs.Q_ext[0].fix(0)  # no longer want external heating in evaporator
    # del m.fs.objective
    # set_up_optimization(m)
    # results = solve(m, solver=solver, tee=False)
    # print("Termination condition: ", results.solver.termination_condition)
    # display_metrics(m)
    # display_design(m)
    # display_corrosion(m)

    # print("\n***---Third solve - optimization with increased brine temperature upper bound---***")
    # m.fs.evaporator.properties_vapor[0].temperature.setub(95 + 273.15)
    # results = solve(m, solver=solver, tee=False)
    # print("Termination condition: ", results.solver.termination_condition)
    # display_metrics(m)
    # display_design(m)
    # display_corrosion(m)


    print("\n***---Fourth solve - optimization with calcite scaling tendency---***")
    add_scaling_tendencies(m)
    assert False # This will stop the code here
    results = solve(m, solver=solver, tee=False)
    print("Termination condition: ", results.solver.termination_condition)
    display_metrics(m)
    display_design(m)
    display_corrosion(m)
    display_scaling_tendencies(m)
    return m, results


def build(material):
    # flowsheet set up
    m = ConcreteModel()
    m.material = material
    m.fs = FlowsheetBlock(dynamic=False)

    # Properties
    m.fs.properties_feed = props_sw.SeawaterParameterBlock()
    m.fs.properties_vapor = props_w.WaterParameterBlock()


    # Unit models
    m.fs.feed = Feed(property_package=m.fs.properties_feed)

    m.fs.pump_feed = Pump(property_package=m.fs.properties_feed)

    m.fs.separator_feed = Separator(
        property_package=m.fs.properties_feed,
        outlet_list=["hx_distillate_cold", "hx_brine_cold"],
        split_basis=SplittingType.totalFlow,
    )

    m.fs.hx_distillate = HeatExchanger(
        hot_side_name="hot",
        cold_side_name="cold",
        hot={"property_package": m.fs.properties_feed, "has_pressure_change": True},
        cold={"property_package": m.fs.properties_feed, "has_pressure_change": True},
        delta_temperature_callback=delta_temperature_chen_callback,
        flow_pattern=HeatExchangerFlowPattern.countercurrent,
    )
    # Set lower bound of approach temperatures
    m.fs.hx_distillate.delta_temperature_in.setlb(0)
    m.fs.hx_distillate.delta_temperature_out.setlb(0)
    m.fs.hx_distillate.area.setlb(10)

    m.fs.hx_brine = HeatExchanger(
        hot_side_name="hot",
        cold_side_name="cold",
        hot={"property_package": m.fs.properties_feed, "has_pressure_change": True},
        cold={"property_package": m.fs.properties_feed, "has_pressure_change": True},
        delta_temperature_callback=delta_temperature_chen_callback,
        flow_pattern=HeatExchangerFlowPattern.countercurrent,
    )
    # Set lower bound of approach temperatures
    m.fs.hx_brine.delta_temperature_in.setlb(0)
    m.fs.hx_brine.delta_temperature_out.setlb(0)
    m.fs.hx_brine.area.setlb(10)

    m.fs.mixer_feed = Mixer(
        property_package=m.fs.properties_feed,
        momentum_mixing_type=MomentumMixingType.equality,
        inlet_list=["hx_distillate_cold", "hx_brine_cold"],
    )
    m.fs.mixer_feed.pressure_equality_constraints[0, 2].deactivate()

    m.fs.evaporator = Evaporator(
        property_package_feed=m.fs.properties_feed,
        property_package_vapor=m.fs.properties_vapor,
    )

    m.fs.compressor = Compressor(property_package=m.fs.properties_vapor)

    m.fs.condenser = Condenser(property_package=m.fs.properties_vapor)

    m.fs.tb_distillate = Translator(
        inlet_property_package=m.fs.properties_vapor,
        outlet_property_package=m.fs.properties_feed,
    )

    # Translator block to convert distillate exiting condenser from water to seawater prop pack
    @m.fs.tb_distillate.Constraint()
    def eq_flow_mass_comp(blk):
        return (
            blk.properties_in[0].flow_mass_phase_comp["Liq", "H2O"]
            == blk.properties_out[0].flow_mass_phase_comp["Liq", "H2O"]
        )

    @m.fs.tb_distillate.Constraint()
    def eq_temperature(blk):
        return blk.properties_in[0].temperature == blk.properties_out[0].temperature

    @m.fs.tb_distillate.Constraint()
    def eq_pressure(blk):
        return blk.properties_in[0].pressure == blk.properties_out[0].pressure

    m.fs.pump_brine = Pump(property_package=m.fs.properties_feed)

    m.fs.pump_distillate = Pump(property_package=m.fs.properties_feed)

    m.fs.distillate = Product(property_package=m.fs.properties_feed)

    m.fs.brine = Product(property_package=m.fs.properties_feed)

    # Connections and connect condenser and evaporator
    m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.pump_feed.inlet)
    m.fs.s02 = Arc(source=m.fs.pump_feed.outlet, destination=m.fs.separator_feed.inlet)
    m.fs.s03 = Arc(
        source=m.fs.separator_feed.hx_distillate_cold,
        destination=m.fs.hx_distillate.cold_inlet,
    )
    m.fs.s04 = Arc(
        source=m.fs.separator_feed.hx_brine_cold, destination=m.fs.hx_brine.cold_inlet
    )
    m.fs.s05 = Arc(
        source=m.fs.hx_distillate.cold_outlet,
        destination=m.fs.mixer_feed.hx_distillate_cold,
    )
    m.fs.s06 = Arc(
        source=m.fs.hx_brine.cold_outlet, destination=m.fs.mixer_feed.hx_brine_cold
    )
    m.fs.s07 = Arc(
        source=m.fs.mixer_feed.outlet, destination=m.fs.evaporator.inlet_feed
    )
    m.fs.s08 = Arc(
        source=m.fs.evaporator.outlet_vapor, destination=m.fs.compressor.inlet
    )
    m.fs.s09 = Arc(source=m.fs.compressor.outlet, destination=m.fs.condenser.inlet)
    m.fs.s10 = Arc(
        source=m.fs.evaporator.outlet_brine, destination=m.fs.pump_brine.inlet
    )
    m.fs.s11 = Arc(source=m.fs.pump_brine.outlet, destination=m.fs.hx_brine.hot_inlet)
    m.fs.s12 = Arc(source=m.fs.hx_brine.hot_outlet, destination=m.fs.brine.inlet)
    m.fs.s13 = Arc(source=m.fs.condenser.outlet, destination=m.fs.tb_distillate.inlet)
    m.fs.s14 = Arc(
        source=m.fs.tb_distillate.outlet, destination=m.fs.pump_distillate.inlet
    )
    m.fs.s15 = Arc(
        source=m.fs.pump_distillate.outlet, destination=m.fs.hx_distillate.hot_inlet
    )
    m.fs.s16 = Arc(
        source=m.fs.hx_distillate.hot_outlet, destination=m.fs.distillate.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m)

    m.fs.evaporator.connect_to_condenser(m.fs.condenser)

    # Add costing
    add_costing(m)

    # Add recovery ratio
    m.fs.recovery = Var(m.fs.config.time, initialize=0.5, bounds=(0, 1))
    m.fs.recovery_equation = Constraint(
        expr=m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"]
        == m.fs.recovery[0]
        * (
            m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"]
            + m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"]
        )
    )

    # Make split ratio equal to recovery
    m.fs.split_ratio_recovery_equality = Constraint(
        expr=m.fs.separator_feed.split_fraction[0, "hx_distillate_cold"]
        == m.fs.recovery[0]
    )

    # Scaling
    # properties
    m.fs.properties_feed.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Liq", "H2O")
    )
    m.fs.properties_feed.set_default_scaling(
        "flow_mass_phase_comp", 1e2, index=("Liq", "TDS")
    )
    m.fs.properties_vapor.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Vap", "H2O")
    )
    m.fs.properties_vapor.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Liq", "H2O")
    )

    # unit model values
    # pumps
    iscale.set_scaling_factor(m.fs.pump_feed.control_volume.work, 1e-3)
    iscale.set_scaling_factor(m.fs.pump_brine.control_volume.work, 1e-3)
    iscale.set_scaling_factor(m.fs.pump_distillate.control_volume.work, 1e-3)

    # distillate HX
    iscale.set_scaling_factor(m.fs.hx_distillate.hot.heat, 1e-3)
    iscale.set_scaling_factor(m.fs.hx_distillate.cold.heat, 1e-3)
    iscale.set_scaling_factor(
        m.fs.hx_distillate.overall_heat_transfer_coefficient, 1e-3
    )

    iscale.set_scaling_factor(m.fs.hx_distillate.area, 1e-1)
    iscale.constraint_scaling_transform(
        m.fs.hx_distillate.cold_side.pressure_balance[0], 1e-5
    )
    iscale.constraint_scaling_transform(
        m.fs.hx_distillate.hot_side.pressure_balance[0], 1e-5
    )

    # brine HX
    iscale.set_scaling_factor(m.fs.hx_brine.hot.heat, 1e-3)
    iscale.set_scaling_factor(m.fs.hx_brine.cold.heat, 1e-3)
    iscale.set_scaling_factor(m.fs.hx_brine.overall_heat_transfer_coefficient, 1e-3)
    iscale.set_scaling_factor(m.fs.hx_brine.area, 1e-1)
    iscale.constraint_scaling_transform(
        m.fs.hx_brine.cold_side.pressure_balance[0], 1e-5
    )
    iscale.constraint_scaling_transform(
        m.fs.hx_brine.hot_side.pressure_balance[0], 1e-5
    )

    # evaporator
    iscale.set_scaling_factor(m.fs.evaporator.area, 1e-3)
    iscale.set_scaling_factor(m.fs.evaporator.U, 1e-3)
    iscale.set_scaling_factor(m.fs.evaporator.delta_temperature_in, 1e-1)
    iscale.set_scaling_factor(m.fs.evaporator.delta_temperature_out, 1e-1)
    iscale.set_scaling_factor(m.fs.evaporator.lmtd, 1e-1)

    # compressor
    iscale.set_scaling_factor(m.fs.compressor.control_volume.work, 1e-6)

    # condenser
    iscale.set_scaling_factor(m.fs.condenser.control_volume.heat, 1e-6)

    # calculate and propagate scaling factors
    iscale.calculate_scaling_factors(m)

    return m


def add_corrosion_rate_surrogate(m):
    # surrogate_dir = f"C:/Users/Carson/idaes/oli-watertap/corrosion_example/surrogate_models/{m.material}/"

    surrogate_dir = f"C:/Users/runak/Documents/Stanford/WE3/WaterTAP/watertap/watertap/flowsheets/mvc/Corrosion_surrogates/stainless_steel_316/"


    # 1. Add indexed version of inputs: temperature, brine salinity, dissolved oxygen
    m.fs.temperature_indexed = Var(
        [0],
        initialize=m.fs.evaporator.properties_brine[0].temperature.value,
        # bounds=(293, 400),
        units=pyunits.dimensionless
    )
    m.fs.eq_temperature_indexed = Constraint(
        expr=m.fs.evaporator.properties_brine[0].temperature == m.fs.temperature_indexed[0] + 273.15
    )

    brine_salt = m.fs.evaporator.properties_brine[0].flow_mass_phase_comp['Liq','TDS'].value
    brine_water = m.fs.evaporator.properties_brine[0].flow_mass_phase_comp['Liq','H2O'].value
    m.fs.brine_salinity_indexed = Var(
        [0],
        initialize=brine_salt/(brine_water + brine_salt),
        units=pyunits.dimensionless
    )
    m.fs.eq_brine_salinity_indexed = Constraint(
        expr=m.fs.evaporator.properties_brine[0].mass_frac_phase_comp['Liq','TDS'] == m.fs.brine_salinity_indexed[0]
    )

    m.fs.dissolved_oxygen_index = Var(
        [0],
        initialize=0,
        units=pyunits.dimensionless
    )
    # 2. Add corrosion rate and potential difference outputs
    m.fs.corrosion_rate_indexed = Var(
        [0],
        initialize=0.1,
        units=pyunits.dimensionless
    )
    m.fs.corrosion_rate = Var(
        initialize=0.1,
        bounds=(0, 0.1), # cannot exceed 0.1
        units=pyunits.m**-3 / pyunits.year
    )
    m.fs.eq_corrosion_rate_indexed = Constraint(
        expr=m.fs.corrosion_rate==m.fs.corrosion_rate_indexed[0]
    )

    m.fs.potential_difference_indexed = Var(
        [0],
        initialize=0.0,
        units=pyunits.dimensionless
    )
    m.fs.potential_difference = Var(
        initialize=0,
        bounds=(0,100),
        units=pyunits.dimensionless
    )
    m.fs.eq_potential_difference_indexed = Constraint(
        expr=m.fs.potential_difference == m.fs.potential_difference_indexed[0]
    )
    # 3. Add corrosion rate surrogate - input order:'Temperature', 'Brine salinity', 'Dissolved oxygen mgO2'
    filename = surrogate_dir + "final_surrogate/corrosion_rate.json"
    corrosion_rate_surrogate = PysmoSurrogate.load_from_file(filename)
    m.fs.corrosion_rate_surrogate = SurrogateBlock(concrete=True)
    m.fs.corrosion_rate_surrogate.build_model(corrosion_rate_surrogate,
                                              input_vars=[m.fs.temperature_indexed[0],
                                                          m.fs.brine_salinity_indexed[0],
                                                          m.fs.dissolved_oxygen_index[0]],
                                              output_vars=[m.fs.corrosion_rate_indexed[0]])
    # check value
    calculate_variable_from_constraint(m.fs.corrosion_rate_indexed[0], m.fs.corrosion_rate_surrogate.pysmo_constraint['Corrosion Rate'])

    # 4. Add potential different surrogate
    filename = surrogate_dir + "final_surrogate/potential_difference.json"
    potential_difference_surrogate = PysmoSurrogate.load_from_file(filename)
    m.fs.potential_difference_surrogate = SurrogateBlock(concrete=True)
    m.fs.potential_difference_surrogate.build_model(potential_difference_surrogate,
                                              input_vars=[m.fs.temperature_indexed[0],
                                                          m.fs.brine_salinity_indexed[0],
                                                          m.fs.dissolved_oxygen_index[0]],
                                              output_vars=[m.fs.potential_difference_indexed[0]])
    # check value
    calculate_variable_from_constraint(m.fs.potential_difference_indexed[0], m.fs.potential_difference_surrogate.pysmo_constraint['Potential Difference'])

def set_surrogate_conditions(m,do=0):
    # fix dissolved oxygen
    m.fs.dissolved_oxygen_index[0].fix(do)
    # fix material factor corresponding to surrogate
    material_factor = {
        "carbon_steel_1018": 1,
        "stainless_steel_304": 3.0,
        "stainless_steel_316": 3.1,
        "duplex_stainless_steel_2205": 3.4,
        "duplex_stainless_steel_2507": 3.5,
        "nickel_alloy_625": 3.9,
        "nickel_alloy_825": 4
    }
    m.fs.costing.evaporator.material_factor_cost.fix(material_factor[m.material])

def add_Q_ext(m, time_point=None):
    # Allows additional heat to be added to evaporator so that an initial feasible solution can be found as a starting
    # guess for optimization in case physically infeasible simulation is proposed

    if time_point is None:
        time_point = m.fs.config.time
    m.fs.Q_ext = Var(time_point, initialize=0, units=pyunits.J / pyunits.s)
    m.fs.Q_ext[0].setlb(0)
    m.fs.evaporator.eq_energy_balance.deactivate()
    m.fs.evaporator.eq_energy_balance_with_additional_Q = Constraint(
        expr=m.fs.evaporator.heat_transfer
        + m.fs.Q_ext[0]
        + m.fs.evaporator.properties_feed[0].enth_flow
        == m.fs.evaporator.properties_brine[0].enth_flow
        + m.fs.evaporator.properties_vapor[0].enth_flow_phase["Vap"]
    )
    iscale.set_scaling_factor(m.fs.Q_ext, 1e-6)


def add_costing(m):
    m.fs.costing = WaterTAPCosting()


    m.fs.pump_feed.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.pump_distillate.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing
    )
    m.fs.pump_brine.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing
    )
    m.fs.hx_distillate.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing
    )
    m.fs.hx_brine.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.mixer_feed.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing
    )
    m.fs.evaporator.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing
    )

    # Pretreatment block
    m.fs.pretreatment = Block()
    # Initialize acid addition flow - removed from add_scaling_tendency function
    m.fs.pretreatment.acid_addition = Var(["HCl"], initialize=0.00001, units=pyunits.mol / pyunits.s)
    #m.fs.pretreatment.acid_addition.display()

    # Define parameter for HCl cost
    m.fs.pretreatment.HCl_cost = Param(initialize=4.7e-3, units=m.fs.costing.base_currency/pyunits.mol, mutable=True)
    # Define expression for flow of HCl
    # m.fs.pretreatment.flow_HCl = Expression(expr=pyunits.convert(m.fs.pretreatment.acid_addition, to_units=pyunits.kg/pyunits.s)) # check unit conversion
    # m.fs.pretreatment.flow_HCl = Expression(
    #     expr=m.fs.pretreatment.acid_addition["HCl"])
    m.fs.pretreatment.flow_HCl = Expression(
        expr=pyunits.convert(m.fs.pretreatment.acid_addition["HCl"], to_units=pyunits.mol / pyunits.s))
    # Register the flow in the costing block
    m.fs.costing.register_flow_type("HCl", m.fs.pretreatment.HCl_cost)
    # Cost_flow method on the costing block
    m.fs.costing.cost_flow(m.fs.pretreatment.flow_HCl, "HCl")


    m.fs.compressor.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing
    )

    m.fs.costing.cost_process()
    m.fs.costing.add_annual_water_production(m.fs.distillate.properties[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.distillate.properties[0].flow_vol)
    m.fs.costing.add_specific_energy_consumption(m.fs.distillate.properties[0].flow_vol)
    m.fs.costing.base_currency = pyo.units.USD_2020

    # m.fs.costing.display()

    # Add costing expressions
    m.fs.costing.MVC_components = Set(initialize=["feed_pump",
                                                  "distillate_pump",
                                                  "brine_pump",
                                                  "hx_distillate",
                                                  "hx_brine",
                                                  "mixer",
                                                  "evaporator",
                                                  "compressor"])
    # Percentage of capital costs
    m.fs.costing.MVC_capital_cost_percentage = Expression(m.fs.costing.MVC_components)
    m.fs.costing.MVC_capital_cost_percentage["feed_pump"] = (
            m.fs.pump_feed.costing.capital_cost / m.fs.costing.aggregate_capital_cost)
    m.fs.costing.MVC_capital_cost_percentage["distillate_pump"] = (
            m.fs.pump_distillate.costing.capital_cost / m.fs.costing.aggregate_capital_cost)
    m.fs.costing.MVC_capital_cost_percentage["brine_pump"] = (
            m.fs.pump_brine.costing.capital_cost / m.fs.costing.aggregate_capital_cost)
    m.fs.costing.MVC_capital_cost_percentage["hx_distillate"] = (
            m.fs.hx_distillate.costing.capital_cost / m.fs.costing.aggregate_capital_cost)
    m.fs.costing.MVC_capital_cost_percentage["hx_brine"] = (
            m.fs.hx_brine.costing.capital_cost / m.fs.costing.aggregate_capital_cost)
    m.fs.costing.MVC_capital_cost_percentage["mixer"] = (
            m.fs.mixer_feed.costing.capital_cost / m.fs.costing.aggregate_capital_cost)
    m.fs.costing.MVC_capital_cost_percentage["evaporator"] = (
            m.fs.evaporator.costing.capital_cost / m.fs.costing.aggregate_capital_cost)
    m.fs.costing.MVC_capital_cost_percentage["compressor"] = (
            m.fs.compressor.costing.capital_cost / m.fs.costing.aggregate_capital_cost)

    # Percentage of costs normalized to LCOW
    m.fs.costing.annual_operating_costs = Expression(
        expr=m.fs.costing.total_capital_cost * m.fs.costing.capital_recovery_factor + m.fs.costing.total_operating_cost)



    m.fs.costing.MVC_LCOW_comp = Set(initialize=["feed_pump",
                                                 "distillate_pump",
                                                 "brine_pump",
                                                 "hx_distillate",
                                                 "hx_brine",
                                                 "mixer",
                                                 "evaporator",
                                                 "compressor",
                                                 "electricity",
                                                 "MLC",
                                                 "capital_costs",
                                                 "operating_costs",
                                                 "capex_opex_ratio",
                                                 "pretreatment"]) # included acid addition
    m.fs.costing.LCOW_percentage = Expression(m.fs.costing.MVC_LCOW_comp)
    m.fs.costing.LCOW_percentage["feed_pump"] = (
            m.fs.pump_feed.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["distillate_pump"] = (
            m.fs.pump_distillate.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["brine_pump"] = (
            m.fs.pump_brine.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["hx_distillate"] = (
            m.fs.hx_distillate.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["hx_brine"] = (
            m.fs.hx_brine.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["mixer"] = (
            m.fs.mixer_feed.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["evaporator"] = (
            m.fs.evaporator.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["compressor"] = (
            m.fs.compressor.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage['electricity'] = (m.fs.costing.aggregate_flow_costs[
                                                       'electricity'] * m.fs.costing.utilization_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage['MLC'] = (
            m.fs.costing.maintenance_labor_chemical_operating_cost / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["capital_costs"] = (
            m.fs.costing.total_capital_cost * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["operating_costs"] = (
            m.fs.costing.total_operating_cost / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage['capex_opex_ratio'] = (
            m.fs.costing.total_capital_cost * m.fs.costing.capital_recovery_factor / m.fs.costing.total_operating_cost)

    # Add acid cost to cost breakdown expressions
    m.fs.costing.LCOW_percentage['pretreatment'] = (m.fs.costing.aggregate_flow_costs[
                                                       'HCl'] * m.fs.costing.utilization_factor / m.fs.costing.annual_operating_costs)


def add_evap_hx_material_factor_equal_constraint(m):
    m.fs.costing.heat_exchanger.material_factor_cost.unfix()
    # make HX material factor equal to evaporator material factor
    m.fs.costing.hx_material_factor_constraint = Constraint(
        expr=m.fs.costing.heat_exchanger.material_factor_cost == m.fs.costing.evaporator.material_factor_cost)
    # m.fs.costing.heat_exchanger.material_factor_cost = m.fs.costing.evaporator.material_factor_cost.value

def add_scaling_tendencies(m):
    # define sea water composition concentration in mg/l and pH
    sea_water_composition = {
        "Na": 10556,
        "K": 380,
        "Ca": 400,
        "Mg": 1262,
        "Cl": 18980,
        "SO4": 2649,
        "HCO3": 140,
    }
    sea_water_pH = 7.56

    """Feed concentration, species mass flow, and pH variables"""
    # Variable (7 because 7 ions): feed species concentration
    ions = list(sea_water_composition.keys())
    m.fs.feed.species_concentrations = Var(
        ions,
        initialize=1,
        units=pyunits.mg / pyunits.L,
    )
    # Variable (8 = 7 ions + 1 H2O): feed species mass flows
    ions.append("H2O")
    m.fs.feed.species_mass_flows = Var(
        ions,
        initialize=1,
        units=pyunits.kg/pyunits.s
    )
    # Variable (1): pH of raw feed (raw --> no acid addition)
    m.fs.feed.pH = Var(initialize=sea_water_pH)

    # Constraint (1): feed TDS is equal to the sum of species mass flow
    m.fs.eq_feed_TDS_constraint = Constraint(
        expr = m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"] == sum(
            m.fs.feed.species_mass_flows[ion]
            for ion in ions
            if ion != "H2O"
        )
    )

    # Constraint (1): connect feed species mass flow of H2O to feed H2O flow rate
    m.fs.eq_feed_H2O_constraint = Constraint(
        expr=m.fs.feed.species_mass_flows['H2O'] == m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"]
    )

    # Constraint (7): 2 to convert feed concentrations to mass flows (based on seawater prop package density)
    @m.fs.feed.Constraint(list(m.fs.feed.species_concentrations.keys()))
    def eq_feed_species_mass_flows(fs, ion):
        #calculate mass flow based on density
        return m.fs.feed.species_mass_flows[ion] == pyunits.convert(
            m.fs.feed.species_concentrations[ion]
            * m.fs.feed.species_mass_flows["H2O"]
            / m.fs.feed.properties[0].dens_mass_phase["Liq"],
            to_units=pyunits.kg / pyunits.s,
        )

    """Evaporator brine species mass flow, pH, and acid addition"""
    # Variable (8): brine species mass flows
    m.fs.evaporator.brine_species_mass_flows = Var(
        ions,
        initialize=1,
        units=pyunits.kg / pyunits.s
    )

    # Variable (1): brine pH
    m.fs.evaporator.brine_pH = Var(initialize=1,units=pyunits.dimensionless)
    m.fs.evaporator.brine_pH.setlb(4)
    # Variable (1): acid addition - removed from here since I add it to the add_costing function
    # m.fs.pretreatment.acid_addition = Var(initialize=0.00001, units=pyunits.mol / pyunits.s)

    # Constraint (8): brine species mass flows
    @m.fs.evaporator.Constraint(list(m.fs.evaporator.brine_species_mass_flows.keys()))
    def eq_feed_to_brine_mass_flows(fs, ion):
        if ion == 'H2O':
            return m.fs.evaporator.brine_species_mass_flows[ion] == m.fs.evaporator.properties_brine[0].flow_mass_phase_comp['Liq', 'H2O']
        else: # species mass balance
            return m.fs.evaporator.brine_species_mass_flows[ion] == m.fs.feed.species_mass_flows[ion]

    # Variable (1): calcite scaling tendency in brine
    m.fs.evaporator.brine_scaling_tendencies = Var(
        [
            ("scalingTendency", "Calcite"),
        ],
        initialize=1,
    )
    print('DOF after RKT variable added but not RKT block: ', degrees_of_freedom(m))
    """ Build reaktoro output dictionary for the scaling tendency of the evaporator brine """
    # add evaporator brine scaling tendencies
    m.fs.evaporator.reaktoro_output_properties = {}
    for key, obj in m.fs.evaporator.brine_scaling_tendencies.items():
        m.fs.evaporator.reaktoro_output_properties[key] = obj
    # add evaporator brine pH
    m.fs.evaporator.reaktoro_output_properties[("pH", None)] = m.fs.evaporator.brine_pH

    """ Configure brine properties reaktoro block """
    m.fs.evaporator.eq_reaktoro_properties = ReaktoroBlock(
        system_state={
            "temperature":m.fs.evaporator.properties_brine[0].temperature,
            "pressure":m.fs.evaporator.properties_brine[0].pressure,
            "pH": m.fs.feed.pH,
        },
        aqueous_phase={
            "composition":m.fs.evaporator.brine_species_mass_flows, # should be based on brine in the evaporator
            "convert_to_rkt_species": True,
            # We can use default converter as its defined for default database (Phreeqc and pitzer)
            "activity_model": reaktoro.ActivityModelPitzer(),  # Can provide a string, or Reaktoro initialized class
            # "fixed_solvent_specie": "H2O",  # We need to define our aqueous solvent as we have to speciate the block
        },
        outputs=m.fs.evaporator.reaktoro_output_properties, # outputs we desired
        chemistry_modifier={"HCl": m.fs.pretreatment.acid_addition["HCl"]},
        build_speciation_block=True,
        database="PhreeqcDatabase",  # can also be reaktoro.PhreeqcDatabase('pitzer.dat')
        database_file="pitzer.dat",  # needs to be a string that names the database file or points to its location
        dissolve_species_in_reaktoro=True,  # This will sum up all species into elements in Reaktoro directly, if set to false, it will build Pyomo constraints instead
        # assert_charge_neutrality=False,  # This is True by Default, but here we actually want to adjust the input speciation till the charge is zero
    )

    """Fix and unfix variables and scaling of variables and constraints"""
    #Unfix feed mass fraction of feed
    m.fs.feed.properties[0].mass_frac_phase_comp['Liq', 'TDS'].unfix()
    # Fix and scale feed pH (1 variable fixed)
    m.fs.feed.pH.fix()
    set_scaling_factor(m.fs.feed.pH, 1)

    # Fix and scale feed species concentrations (7 variables fixed)
    for ion, value in sea_water_composition.items():
        m.fs.feed.species_concentrations[ion].fix(value)
        set_scaling_factor(m.fs.feed.species_concentrations[ion], 1 / value)

    # Fix pH first
    # m.fs.evaporator.brine_pH.fix(6.8)
    set_scaling_factor(m.fs.evaporator.brine_pH, 1)
    # Scale acid addition
    m.fs.pretreatment.acid_addition["HCl"].fix()
    set_scaling_factor(m.fs.pretreatment.acid_addition["HCl"], 1 / 0.001)

    """Initialization"""
    # initialize mass flow constraints for raw feed
    for comp, pyoobj in m.fs.feed.eq_feed_species_mass_flows.items():
        calculate_variable_from_constraint(
            m.fs.feed.species_mass_flows[comp], pyoobj
        )
        set_scaling_factor(
            m.fs.feed.species_mass_flows[comp],
            1 / m.fs.feed.species_mass_flows[comp].value,
        )
        constraint_scaling_transform(
            pyoobj, 1 / m.fs.feed.species_mass_flows[comp].value
        )
    # Initialize water flow constraints for feed
    set_scaling_factor(m.fs.feed.species_mass_flows["H2O"], 1)
    m.fs.feed.species_mass_flows['H2O'] = m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'H2O']

    # initialize concentration constraints for evaporator brine
    for comp, pyoobj in m.fs.evaporator.eq_feed_to_brine_mass_flows.items():
        if "H2O" in comp:
            set_scaling_factor(m.fs.evaporator.brine_species_mass_flows[comp], 1)
            calculate_variable_from_constraint(
                m.fs.evaporator.brine_species_mass_flows[comp], pyoobj
            )
        else:
            calculate_variable_from_constraint(
                m.fs.evaporator.brine_species_mass_flows[comp], pyoobj
            )
            set_scaling_factor(
                m.fs.evaporator.brine_species_mass_flows[comp],
                1 / m.fs.evaporator.brine_species_mass_flows[comp].value,
            )
            constraint_scaling_transform(
                pyoobj, 1 / m.fs.evaporator.brine_species_mass_flows[comp].value
            )

    # initialize evaporator reaktoro block
    m.fs.evaporator.eq_reaktoro_properties.initialize()

    # Check DOF
    print("DOFs:", degrees_of_freedom(m)) # DOF is going to be 4 + 2 because we already ran optimization
    outputs_main_block = len(m.fs.evaporator.eq_reaktoro_properties.reaktoro_model.outputs)
    print("Number of Reaktoro outputs", outputs_main_block)
    print(
        "Actual DOFs:",
        degrees_of_freedom(m) - (outputs_main_block),
    )
    # assert degrees_of_freedom(m) - (outputs_main_block) == 0

def solve_with_reaktoro(m, solver=None):
    # run solver
    if solver is None:
        solver = get_solver(solver="cyipopt-watertap")
        solver.options["max_iter"] = 100

    result = solver.solve(m, tee=True)
    assert_optimal_termination(result)

    #dsiplay evaporator brine reaktoro block outputs
    # m.fs.evaporator.eq_reaktoro_properties.outputs.display()

def setup_optimization_with_reaktoro(m, solver=None):
    m.fs.evaporator.reaktoro_output_properties[("scalingTendency", "Calcite")].setub(1)
    m.fs.pretreatment.acid_addition["HCl"].unfix()

def set_operating_conditions(m):
    # Feed inlet
    m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"].fix(0.1)
    m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].fix(40)
    # m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].fix(4)
    m.fs.feed.properties[0].temperature.fix(273.15 + 25)
    m.fs.feed.properties[0].pressure.fix(101325)

    m.fs.recovery[0].fix(0.5)

    # Feed pump
    m.fs.pump_feed.efficiency_pump[0].fix(0.8)
    m.fs.pump_feed.control_volume.deltaP[0].fix(7e3)

    # Separator
    m.fs.separator_feed.split_fraction[0, "hx_distillate_cold"] = m.fs.recovery[0].value

    # Distillate HX
    m.fs.hx_distillate.overall_heat_transfer_coefficient[0].fix(2e3)
    m.fs.hx_distillate.area.fix(125)
    m.fs.hx_distillate.cold.deltaP[0].fix(7e3)
    m.fs.hx_distillate.hot.deltaP[0].fix(7e3)

    # Brine HX
    m.fs.hx_brine.overall_heat_transfer_coefficient[0].fix(2e3)
    m.fs.hx_brine.area.fix(115)
    m.fs.hx_brine.cold.deltaP[0].fix(7e3)
    m.fs.hx_brine.hot.deltaP[0].fix(7e3)

    # Evaporator
    m.fs.evaporator.inlet_feed.temperature[0] = 50 + 273.15  # provide guess
    m.fs.evaporator.outlet_brine.temperature[0].fix(70 + 273.15)
    m.fs.evaporator.U.fix(3e3)  # W/K-m^2
    m.fs.evaporator.area.setub(1e4)  # m^2

    # Compressor
    m.fs.compressor.pressure_ratio.fix(1.6)
    m.fs.compressor.efficiency.fix(0.8)

    # Brine pump
    m.fs.pump_brine.efficiency_pump[0].fix(0.8)
    m.fs.pump_brine.control_volume.deltaP[0].fix(4e4)

    # Distillate pump
    m.fs.pump_distillate.efficiency_pump[0].fix(0.8)
    m.fs.pump_distillate.control_volume.deltaP[0].fix(4e4)

    # Fix 0 TDS
    m.fs.tb_distillate.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].fix(1e-5)

    # Costing
    m.fs.costing.TIC.fix(2)
    m.fs.costing.electricity_cost = 0.1  # 0.15
    m.fs.costing.heat_exchanger.material_factor_cost.fix(5)
    m.fs.costing.evaporator.material_factor_cost.fix(5)
    m.fs.costing.compressor.unit_cost.fix(1 * 7364)

    # Temperature bounds
    m.fs.evaporator.properties_vapor[0].temperature.setub(75 + 273.15)
    m.fs.compressor.control_volume.properties_out[0].temperature.setub(450)

    # check degrees of freedom
    print("DOF after setting operating conditions: ", degrees_of_freedom(m))


def initialize_system(m, solver=None):
    if solver is None:
        solver = get_solver()
    optarg = solver.options

    # Touch feed mass fraction property
    m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"]
    solver.solve(m.fs.feed)

    # Propagate vapor flow rate based on given recovery
    m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp[
        "Vap", "H2O"
    ] = m.fs.recovery[0] * (
        m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"]
        + m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"]
    )
    m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Liq", "H2O"] = 0

    # Propagate brine salinity and flow rate
    m.fs.evaporator.properties_brine[0].mass_frac_phase_comp["Liq", "TDS"] = (
        m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"]
        / (1 - m.fs.recovery[0])
    )
    m.fs.evaporator.properties_brine[0].mass_frac_phase_comp["Liq", "H2O"] = (
        1 - m.fs.evaporator.properties_brine[0].mass_frac_phase_comp["Liq", "TDS"].value
    )
    m.fs.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "TDS"] = (
        m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"]
    )
    m.fs.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "H2O"] = (
        m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"]
        - m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"]
    )

    # initialize feed pump
    propagate_state(m.fs.s01)
    m.fs.pump_feed.initialize(optarg=optarg, solver="ipopt-watertap")

    # initialize separator
    propagate_state(m.fs.s02)
    # Touch property for initialization
    m.fs.separator_feed.mixed_state[0].mass_frac_phase_comp["Liq", "TDS"]
    m.fs.separator_feed.split_fraction[0, "hx_distillate_cold"].fix(
        m.fs.recovery[0].value
    )
    m.fs.separator_feed.mixed_state.initialize(optarg=optarg, solver="ipopt-watertap")
    # Touch properties for initialization
    m.fs.separator_feed.hx_brine_cold_state[0].mass_frac_phase_comp["Liq", "TDS"]
    m.fs.separator_feed.hx_distillate_cold_state[0].mass_frac_phase_comp["Liq", "TDS"]
    m.fs.separator_feed.initialize(optarg=optarg, solver="ipopt-watertap")
    m.fs.separator_feed.split_fraction[0, "hx_distillate_cold"].unfix()

    # initialize distillate heat exchanger
    propagate_state(m.fs.s03)
    m.fs.hx_distillate.cold_outlet.temperature[0] = (
        m.fs.evaporator.inlet_feed.temperature[0].value
    )
    m.fs.hx_distillate.cold_outlet.pressure[0] = m.fs.evaporator.inlet_feed.pressure[
        0
    ].value
    m.fs.hx_distillate.hot_inlet.flow_mass_phase_comp[0, "Liq", "H2O"] = (
        m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].value
    )
    m.fs.hx_distillate.hot_inlet.flow_mass_phase_comp[0, "Liq", "TDS"] = 1e-4
    m.fs.hx_distillate.hot_inlet.temperature[0] = (
        m.fs.evaporator.outlet_brine.temperature[0].value
    )
    m.fs.hx_distillate.hot_inlet.pressure[0] = 101325
    m.fs.hx_distillate.initialize(solver="ipopt-watertap")

    # initialize brine heat exchanger
    propagate_state(m.fs.s04)
    m.fs.hx_brine.cold_outlet.temperature[0] = m.fs.evaporator.inlet_feed.temperature[
        0
    ].value
    m.fs.hx_brine.cold_outlet.pressure[0] = m.fs.evaporator.inlet_feed.pressure[0].value
    m.fs.hx_brine.hot_inlet.flow_mass_phase_comp[0, "Liq", "H2O"] = (
        m.fs.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "H2O"]
    )
    m.fs.hx_brine.hot_inlet.flow_mass_phase_comp[0, "Liq", "TDS"] = (
        m.fs.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "TDS"]
    )
    m.fs.hx_brine.hot_inlet.temperature[0] = m.fs.evaporator.outlet_brine.temperature[
        0
    ].value
    m.fs.hx_brine.hot_inlet.pressure[0] = 101325
    m.fs.hx_brine.initialize(solver="ipopt-watertap")

    # initialize mixer
    propagate_state(m.fs.s05)
    propagate_state(m.fs.s06)
    m.fs.mixer_feed.initialize(solver="ipopt-watertap")
    m.fs.mixer_feed.pressure_equality_constraints[0, 2].deactivate()

    # initialize evaporator
    propagate_state(m.fs.s07)
    m.fs.Q_ext[0].fix()
    m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].fix()
    # fixes and unfixes those values
    m.fs.evaporator.initialize(delta_temperature_in=60, solver="ipopt-watertap")
    m.fs.Q_ext[0].unfix()
    m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].unfix()

    # initialize compressor
    propagate_state(m.fs.s08)
    m.fs.compressor.initialize(solver="ipopt-watertap")

    # initialize condenser
    propagate_state(m.fs.s09)
    m.fs.condenser.initialize(
        heat=-m.fs.evaporator.heat_transfer.value, solver="ipopt-watertap"
    )

    # initialize brine pump
    propagate_state(m.fs.s10)
    m.fs.pump_brine.initialize(optarg=optarg, solver="ipopt-watertap")

    # initialize distillate pump
    propagate_state(m.fs.s13)  # to translator block
    propagate_state(m.fs.s14)  # from translator block to pump
    m.fs.pump_distillate.control_volume.properties_in[0].temperature = (
        m.fs.condenser.control_volume.properties_out[0].temperature.value
    )
    m.fs.pump_distillate.control_volume.properties_in[0].pressure = (
        m.fs.condenser.control_volume.properties_out[0].pressure.value
    )
    m.fs.pump_distillate.initialize(optarg=optarg, solver="ipopt-watertap")

    # propagate brine state
    propagate_state(m.fs.s12)
    propagate_state(m.fs.s16)

    seq = SequentialDecomposition(tear_solver="cbc")
    seq.options.log_info = False
    seq.options.iterLim = 5

    def func_initialize(unit):
        if unit.local_name == "feed":
            pass
        elif unit.local_name == "condenser":
            unit.initialize(
                heat=-unit.flowsheet().evaporator.heat_transfer.value,
                optarg=solver.options,
                solver="ipopt-watertap",
            )
        elif unit.local_name == "evaporator":
            unit.flowsheet().Q_ext[0].fix()
            unit.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].fix()
            unit.initialize(delta_temperature_in=60, solver="ipopt-watertap")
            unit.flowsheet().Q_ext[0].unfix()
            unit.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].unfix()
        elif unit.local_name == "separator_feed":
            unit.split_fraction[0, "hx_distillate_cold"].fix(
                unit.flowsheet().recovery[0].value
            )
            unit.initialize(solver="ipopt-watertap")
            unit.split_fraction[0, "hx_distillate_cold"].unfix()
        elif unit.local_name == "mixer_feed":
            unit.initialize(solver="ipopt-watertap")
            unit.pressure_equality_constraints[0, 2].deactivate()
        else:
            unit.initialize(solver="ipopt-watertap")

    seq.run(m, func_initialize)

    m.fs.costing.initialize()

    solver.solve(m, tee=False)

    print("Initialization done")


def fix_outlet_pressures(m):
    # The distillate outlet pressure remains unfixed so that there is not an implicit upper bound on the compressed vapor pressure

    # Unfix pump heads
    m.fs.pump_brine.control_volume.deltaP[0].unfix()
    # m.fs.pump_distillate.control_volume.deltaP[0].unfix()

    # Fix outlet pressures
    m.fs.brine.properties[0].pressure.fix(101325)
    # m.fs.distillate.properties[0].pressure.fix(101325)

    return


def calculate_cost_sf(cost):
    sf = 10 ** -(math.log10(abs(cost.value)))
    iscale.set_scaling_factor(cost, sf)


def scale_costs(m):
    calculate_cost_sf(m.fs.hx_distillate.costing.capital_cost)
    calculate_cost_sf(m.fs.hx_brine.costing.capital_cost)
    calculate_cost_sf(m.fs.mixer_feed.costing.capital_cost)
    calculate_cost_sf(m.fs.evaporator.costing.capital_cost)
    calculate_cost_sf(m.fs.compressor.costing.capital_cost)
    calculate_cost_sf(m.fs.costing.aggregate_capital_cost)
    calculate_cost_sf(m.fs.costing.aggregate_flow_costs["electricity"])
    calculate_cost_sf(m.fs.costing.total_capital_cost)
    calculate_cost_sf(m.fs.costing.total_operating_cost)
    # calculate_cost_sf(m.fs.evaporator.costing.acid_addition_cost) # acid addition pretreatment

    iscale.calculate_scaling_factors(m)

    print("Scaled costs")


def solve(model, solver=None, tee=False, raise_on_failure=False):
    # ---solving---
    if solver is None:
        solver = get_solver()

    results = solver.solve(model, tee=tee)
    if check_optimal_termination(results):
        return results
    msg = (
        "The current configuration is infeasible. Please adjust the decision variables."
    )
    if raise_on_failure:
        raise RuntimeError(msg)
    else:
        print(msg)
        return results


def set_up_optimization(m):
    m.fs.objective = Objective(expr=m.fs.costing.LCOW)
    m.fs.Q_ext[0].fix(0)
    m.fs.evaporator.area.unfix()
    m.fs.evaporator.outlet_brine.temperature[0].unfix()
    m.fs.compressor.pressure_ratio.unfix()
    m.fs.hx_distillate.area.unfix()
    m.fs.hx_brine.area.unfix()

    print("DOF for optimization: ", degrees_of_freedom(m))


def display_metrics(m):
    print("\nSystem metrics")
    print(
        "Feed flow rate:                           %.2f kg/s"
        % (
            m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value
            + m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
        )
    )
    print(
        "Feed salinity:                            %.2f g/kg"
        % (m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"].value * 1e3)
    )
    print(
        "Brine salinity:                           %.2f g/kg"
        % (
            m.fs.evaporator.properties_brine[0].mass_frac_phase_comp["Liq", "TDS"].value
            * 1e3
        )
    )
    print(
        "Product flow rate:                        %.2f kg/s"
        % m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].value
    )
    print(
        "Recovery:                                 %.2f %%"
        % (m.fs.recovery[0].value * 100)
    )
    print(
        "Specific energy consumption:              %.2f kWh/m3"
        % value(m.fs.costing.specific_energy_consumption)
    )
    print(
        "Levelized cost of water:                  %.2f $/m3" % value(m.fs.costing.LCOW)
    )
    print(
        "External Q:                               %.2f W" % m.fs.Q_ext[0].value
    )  # should be 0 for optimization


def display_design(m):
    print("\nState variables")
    print(
        "Preheated feed temperature:               %.2f K"
        % m.fs.evaporator.properties_feed[0].temperature.value
    )
    print(
        "Evaporator (brine, vapor) temperature:    %.2f K"
        % m.fs.evaporator.properties_brine[0].temperature.value
    )
    print(
        "Evaporator (brine, vapor) pressure:       %.2f kPa"
        % (m.fs.evaporator.properties_vapor[0].pressure.value * 1e-3)
    )
    print(
        "Compressed vapor temperature:             %.2f K"
        % m.fs.compressor.control_volume.properties_out[0].temperature.value
    )
    print(
        "Compressed vapor pressure:                %.2f kPa"
        % (m.fs.compressor.control_volume.properties_out[0].pressure.value * 1e-3)
    )
    print(
        "Condensed vapor temperature:              %.2f K"
        % m.fs.condenser.control_volume.properties_out[0].temperature.value
    )

    print("\nDesign variables")
    print(
        "Brine heat exchanger area:                %.2f m2" % m.fs.hx_brine.area.value
    )
    print(
        "Distillate heat exchanger area:           %.2f m2"
        % m.fs.hx_distillate.area.value
    )
    print(
        "Compressor pressure ratio:                %.2f"
        % m.fs.compressor.pressure_ratio.value
    )
    print(
        "Evaporator area:                          %.2f m2" % m.fs.evaporator.area.value
    )
    print(
        "Evaporator LMTD:                          %.2f K" % m.fs.evaporator.lmtd.value
    )
    print(
        "Evaporator material factor:               %.2f " % m.fs.costing.evaporator.material_factor_cost.value
    )

def display_corrosion(m):
    print('\nCorrosion results')
    print(f'Material:                        {m.material}')
    print(
        "Corrosion rate:                     %.2f mm/yr"
        % m.fs.corrosion_rate.value
    )
    print(
        "Potential difference:                     %.2f V"
        % m.fs.potential_difference.value
    )

def display_scaling_tendencies(m):
    print("\nScaling tendencies")
    for key, obj in m.evaporator.scaling_tendencies.items():
        print(f"{key}, {obj.value}")


def display_reaktoro_metrics(m):
    print('Feed pH: ', m.fs.feed.pH.value)
    print('Brine pH: ', m.fs.evaporator.brine_pH.value)
    print('Acid addition: ', m.fs.pretreatment.acid_addition["HCl"].value)
    print('Calcite scaling tendency: ', m.fs.evaporator.brine_scaling_tendencies[('scalingTendency', 'Calcite')].value)
def feed_salinity_recovery_sweep(material='stainless_steel_316',
                                 do=0):
    save_dir = "C:/Users/Carson/idaes/NAWI-analysis/analysis_waterTAP/analysisWaterTAP/analysis_scripts/mvc_corrosion/results"
    filename = save_dir + f"/{material}_{do}_feed_recovery.csv"

    # # build model
    # m = mvc_setup.build(material=material)
    # print("\ninitialization")
    # mvc_setup.initialize(m, do=do)
    # print("\ninitial optimizing")
    m = build(material=material)
    set_operating_conditions(m)
    add_Q_ext(m, time_point=m.fs.config.time)
    initialize_system(m)
    # rescale costs after initialization because scaling depends on flow rates
    scale_costs(m)
    fix_outlet_pressures(m)  # outlet pressure are initially unfixed for initialization

    # set up for minimizing Q_ext in first solve
    # should be 1 DOF because Q_ext is unfixed
    # print("DOF after initialization: ", degrees_of_freedom(m))
    m.fs.objective = Objective(expr=m.fs.Q_ext[0])

    print("\n***---First solve - simulation results---***")
    solver = get_solver()
    results = solve(m, solver=solver, tee=False)
    print("Termination condition: ", results.solver.termination_condition)
    display_metrics(m)
    display_design(m)

    print("\n***---Second solve - optimization with corrosion rate surrogate---***")
    add_evap_hx_material_factor_equal_constraint(m)
    add_corrosion_rate_surrogate(m)
    set_surrogate_conditions(m, do=do)
    m.fs.Q_ext[0].fix(0)  # no longer want external heating in evaporator
    del m.fs.objective
    set_up_optimization(m)
    results = solve(m, solver=solver, tee=False)
    print("Termination condition: ", results.solver.termination_condition)
    display_metrics(m)
    display_design(m)
    display_corrosion(m)

    print('\n---------\nInitialization DONE\n---------')
    salinity_recovery_dict = {}
    salinity_recovery_dict[25] = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    salinity_recovery_dict[50] = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    salinity_recovery_dict[75] = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    salinity_recovery_dict[100] = [0.4, 0.45, 0.5, 0.55, 0.6]
    salinity_recovery_dict[125] = [0.4, 0.45, 0.5]
    salinity_recovery_dict[150] = [0.4, 0.45, 0.5]

    results_dict = build_results_dict()
    for sal, rec in salinity_recovery_dict.items():
        # start at first recovery
        m.fs.recovery[0].fix(rec[0])
        results = solve(m)
        # print('First recovery termination condition: ', results.solver.termination_condition)
        # now update salinity
        m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"].fix(sal/1000)
        results = solve(m)
        print('Next salinity termination condition:', results.solver.termination_condition)
        for r in rec:
            m.fs.recovery[0].fix(r)
            try:
                results = solve(m)
                results_dict['Termination condition'].append(results.solver.termination_condition)
                update_results_dict(m, results_dict)
            except:
                results_dict['Feed salinity'].append(sal)
                results_dict['Recovery'].append(r)
                results_dict['Material'].append(material)
                results_dict['Termination condition'].append('bad status')
                update_results_dict_error(results_dict)

        # temporarily save results
        results_df = pd.DataFrame(results_dict)
        results_df.to_csv(filename, index=False)

    # save results as dataframe
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(filename, index=False)
    print(f'Saved {material} and {do} dissolved oxygen')

    return

def build_results_dict():
    res_dict = {}
    res_dict['Feed salinity'] = []
    res_dict['Recovery'] = []
    res_dict['Material'] = []
    res_dict['Evaporator temperature'] = []
    res_dict['Feed flow rate'] = []
    res_dict['Brine salinity'] = []
    res_dict['Product flow rate'] = []
    res_dict['SEC'] = []
    res_dict['LCOW'] = []
    res_dict['External Q'] = []
    res_dict['Preheated feed temperature'] = []
    res_dict['Evaporator vapor pressure'] = []
    res_dict['Compressed vapor temperature'] = []
    res_dict['Compressed vapor pressure'] = []
    res_dict['Condensed vapor temperature'] = []
    res_dict['Compressor pressure ratio'] = []
    res_dict['Evaporator area'] = []
    res_dict['Evaporator LMTD'] = []
    res_dict['Evaporator material factor'] = []
    res_dict['Corrosion rate'] = []
    res_dict['Potential difference'] = []
    res_dict['Dissolved oxygen']= []
    res_dict['capex_opex_ratio'] = []
    res_dict['Termination condition'] = []

    return res_dict

def update_results_dict(m, res_dict):
    res_dict['Feed salinity'].append(m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"].value*1000)
    res_dict['Recovery'].append(m.fs.recovery[0].value)
    res_dict['Material'].append(m.material)
    res_dict['Evaporator temperature'].append(m.fs.evaporator.properties_brine[0].temperature.value)
    res_dict['Feed flow rate'].append(m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value +
                                      m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value)
    res_dict['Brine salinity'].append(m.fs.evaporator.properties_brine[0].mass_frac_phase_comp["Liq", "TDS"].value * 1e3)
    res_dict['Product flow rate'].append(m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].value)
    res_dict['SEC'].append(value(m.fs.costing.specific_energy_consumption))
    res_dict['LCOW'].append(value(m.fs.costing.LCOW))
    res_dict['External Q'].append(value(m.fs.Q_ext[0]))
    res_dict['Preheated feed temperature'].append(m.fs.evaporator.properties_feed[0].temperature.value)
    res_dict['Evaporator vapor pressure'].append(m.fs.evaporator.properties_vapor[0].pressure.value)
    res_dict['Compressed vapor temperature'].append(m.fs.compressor.control_volume.properties_out[0].temperature.value)
    res_dict['Compressed vapor pressure'].append(m.fs.compressor.control_volume.properties_out[0].pressure.value)
    res_dict['Condensed vapor temperature'].append(m.fs.condenser.control_volume.properties_out[0].temperature.value)
    res_dict['Compressor pressure ratio'].append(m.fs.compressor.pressure_ratio.value)
    res_dict['Evaporator area'].append(m.fs.evaporator.area.value)
    res_dict['Evaporator LMTD'].append(m.fs.evaporator.lmtd.value)
    res_dict['Evaporator material factor'].append(m.fs.costing.evaporator.material_factor_cost.value)
    res_dict['Corrosion rate'].append(m.fs.corrosion_rate.value)
    res_dict['Potential difference'].append(m.fs.potential_difference.value)
    res_dict['Dissolved oxygen'].append(m.fs.dissolved_oxygen_index[0].value)
    res_dict['capex_opex_ratio'].append(value(m.fs.costing.LCOW_percentage['capex_opex_ratio']))

    return res_dict

def update_results_dict_error(res_dict):
    res_dict['Evaporator temperature'].append(np.NAN)
    res_dict['Feed flow rate'].append(np.NAN)
    res_dict['Brine salinity'].append(np.NAN)
    res_dict['Product flow rate'].append(np.NAN)
    res_dict['SEC'].append(np.NAN)
    res_dict['LCOW'].append(np.NAN)
    res_dict['External Q'].append(np.NAN)
    res_dict['Preheated feed temperature'].append(np.NAN)
    res_dict['Evaporator vapor pressure'].append(np.NAN)
    res_dict['Compressed vapor temperature'].append(np.NAN)
    res_dict['Compressed vapor pressure'].append(np.NAN)
    res_dict['Condensed vapor temperature'].append(np.NAN)
    res_dict['Compressor pressure ratio'].append(np.NAN)
    res_dict['Evaporator area'].append(np.NAN)
    res_dict['Evaporator LMTD'].append(np.NAN)
    res_dict['Evaporator material factor'].append(np.NAN)
    res_dict['Corrosion rate'].append(np.NAN)
    res_dict['Potential difference'].append(np.NAN)
    res_dict['Dissolved oxygen'].append(np.NAN)
    res_dict['capex_opex_ratio'].append(np.NAN)

    return res_dict

if __name__ == "__main__":
    # m = main()

    mat_list = [  # 'carbon_steel_1018',
        # 'stainless_steel_304',
        'stainless_steel_316',
        # 'duplex_stainless_steel_2205',
        # 'duplex_stainless_steel_2507',
        # 'nickel_alloy_625',
        # 'nickel_alloy_825'
        ]
    do_list = [0, 3.5, 7]
    single_run(material='stainless_steel_316', do=0)
    # for mat in mat_list:
    #     for do in do_list:
    #         feed_salinity_recovery_sweep(mat, do)