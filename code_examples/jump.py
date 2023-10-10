# Import necessary Julia packages
from julia import Base, Main

# Define and solve the optimization problem in Julia
Main.eval("""
using JuMP
using GLPK

model = Model(GLPK.Optimizer)
@variable(model, 0 <= x <= 5)
@variable(model, 0 <= y <= 10)
@objective(model, Max, 2x + 5y)
@constraint(model, 1x + 5y <= 10)

optimize!(model)

# Store the results in Julia variables with a Julia module prefix
Main.objective_value_julia = JuMP.objective_value(model)
Main.x_value_julia = JuMP.value(x)
Main.y_value_julia = JuMP.value(y)
""")

# Access the results in Python
objective_value = Main.objective_value_julia
x_value = Main.x_value_julia
y_value = Main.y_value_julia

print("Objective value:", objective_value)
print("x:", x_value)
print("y:", y_value)

# Create a LinPy model
model = Model(solver=GLPK())

# Create variables
x = Variable(lowBound=0, upBound=5)
y = Variable(lowBound=0, upBound=10)

# Add objective function
objective = Objective(2*x + 5*y, MAXIMIZE)
model += objective

# Add constraint
constraint = Constraint(1*x + 5*y, '<=', 10)
model += constraint

# Solve the model
model.optimize()

# Retrieve the results
objective_value = model.objective.value
x_value = x.value
y_value = y.value

# Print the results
print("Objective value:", objective_value)
print("x:", x_value)
print("y:", y_value)
