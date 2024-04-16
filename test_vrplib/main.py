import vrplib
import json
import numpy as np

# Adjusted function to handle objects already in dict format and NumPy arrays
def obj_to_dict(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Process each key-value pair in the dictionary
        return {k: obj_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [obj_to_dict(v) for v in obj]
    else:
        return obj

# Assuming the read_instance and read_solution return dictionaries
# Read VRPLIB formatted instances and solutions
instance = vrplib.read_instance("./Vrp-Set-XML100/instances/XML100_1111_01.vrp")
solutions = vrplib.read_solution("./Vrp-Set-XML100/solutions/XML100_1111_01.sol")

# Convert instance and solutions, handling any nested NumPy arrays
instance_dict = obj_to_dict(instance)
solutions_dict = obj_to_dict(solutions)

# Serialize to JSON
instance_json = json.dumps(instance_dict, indent=4)
solutions_json = json.dumps(solutions_dict, indent=4)

# Print or save to a file
print(instance_json)
print(solutions_json)

# Optionally, save the JSON to a file
with open('instance.json', 'w') as f:
    f.write(instance_json)

with open('solutions.json', 'w') as f:
    f.write(solutions_json)
