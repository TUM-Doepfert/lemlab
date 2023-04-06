ex = {
    'admin': [0, ['database', 'admin']],
}

# Define the source dictionary
database = [    {'admin': {'name': 'Alice', 'age': 30}},    {'admin': {'name': 'Bob', 'age': 40}}]

# Extract the information using the specified keys
position, keys = ex['admin']
info = database[position]
for key in keys:
    info = info[key]

# Print the result
print(info)  # Output: {'name': 'Alice', 'age': 30}
