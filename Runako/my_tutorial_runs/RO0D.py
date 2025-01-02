# Import libraries
# Imports from Pyomo, including "value" for getting the
# value of Pyomo objects
from pyomo.environ import ConcreteModel, Objective, Expression, value

# Imports from IDAES
# Import from flowsheet block from IDAES core
from idaes.core import FlowsheetBlock
# Import function to get default solver
from watertap.core.solvers import get_solver
# Import function to check degrees of freedom
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.scaling import calculate_scaling_factors, set_scaling_factor

# Imports from WaterTAP
# Import NaCl property model
from watertap.property_models.NaCl_prop_pack import NaClParameterBlock
# Import RO model
from watertap.unit_models.reverse_osmosis_0D import (ReverseOsmosis0D,
        ConcentrationPolarizationType, MassTransferCoefficient)

# Start building the RO model

# Create a Pyomo concrete model, flowsheet, and NaCl property parameter block.
m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
m.fs.properties = NaClParameterBlock()

# Add an RO unit to the flowsheet.
m.fs.unit = ReverseOsmosis0D(
    property_package=m.fs.properties,
    concentration_polarization_type=ConcentrationPolarizationType.none,
    mass_transfer_coefficient=MassTransferCoefficient.none,
    has_pressure_change=False,
    )

# Step 3: Specify values for system variables

m.fs.unit.inlet.flow_mass_phase_comp[0, 'Liq', 'NaCl'].fix(0.035)  # mass flow rate of NaCl (kg/s)
m.fs.unit.inlet.flow_mass_phase_comp[0, 'Liq', 'H2O'].fix(0.965)   # mass flow rate of water (kg/s)
m.fs.unit.inlet.pressure[0].fix(50e5)                              # feed pressure (Pa)
m.fs.unit.inlet.temperature[0].fix(298.15)                         # feed temperature (K)
m.fs.unit.area.fix(50)                                             # membrane area (m^2)
m.fs.unit.A_comp.fix(4.2e-12)                                      # membrane water permeability (m/Pa/s)
m.fs.unit.B_comp.fix(3.5e-8)                                       # membrane salt permeability (m/s)
m.fs.unit.permeate.pressure[0].fix(101325)                         # permeate pressure (Pa)

# Step 4: Scale all variables

# Set scaling factors for component mass flowrates.
m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1, index=('Liq', 'H2O'))
m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1e2, index=('Liq', 'NaCl'))

# Set scaling factor for membrane area.
set_scaling_factor(m.fs.unit.area, 1e-2)

# Calculate scaling factors for all other variables.
calculate_scaling_factors(m)

# Step 5: Initialize the model

m.fs.unit.initialize()

# Step 6: Setup a solver and run a simulation

# Check that degrees of freedom = 0 before attempting simulation.
# This means that the performance of the flowsheet is completely
# determined by the system variables that were fixed above.
assert degrees_of_freedom(m) == 0

# Setup solver
solver = get_solver()

# Run simulation
simulation_results = solver.solve(m)

# Display report, reports include a small subset of the most important variables
m.fs.unit.report()

# Display all results, this shows all variables and constraints
m.fs.unit.display()

# Step 7: Unfix variables, set variable bounds, and run optimization to minimize specific energy consumption.

# Unfix membrane area and feed pressure
m.fs.unit.area.unfix()                  # membrane area (m^2)
m.fs.unit.inlet.pressure[0].unfix()     # feed pressure (Pa)

# Set lower and upper bounds for membrane area (m^2)
m.fs.unit.area.setlb(1)
m.fs.unit.area.setub(500)

# Set lower and upper bounds for feed pressure (Pa)
m.fs.unit.inlet.pressure[0].setlb(10e5)
m.fs.unit.inlet.pressure[0].setub(80e5)

# Assume 100% efficiency of pumps and ERD and no pressure losses
#--> Pump power consumption ~ Qp*Pf/3.6e6
m.fs.specific_energy_consumption = Expression(
    expr=m.fs.unit.inlet.pressure[0]/(3.6e6))

# Define objective function to minimize the specific energy consumption.
m.fs.objective = Objective(expr=m.fs.specific_energy_consumption)

# Set the water recovery to 50%
m.fs.unit.recovery_vol_phase[0,'Liq'].fix(0.50)

# The solver will find the membrane area and
# inlet pressure that achieve 50% recovery while minimizing
# specific energy consumption. Since we fixed the
# volumetric water recovery, a degree of freedom
# was removed from the model and is now 1.
print(degrees_of_freedom(m))

# Solve the model
optimization_results = solver.solve(m)
print(optimization_results)

# membrane area of the optimized RO unit
value(m.fs.unit.area)

# inlet pressure of the optimized RO unit
value(m.fs.unit.inlet.pressure[0])

# the minimum specific energy consumption
value(m.fs.specific_energy_consumption)
# display the overall report on the RO unit
m.fs.unit.report()