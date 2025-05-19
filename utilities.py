import cdflib

def print_info(cdf):

    for key, value in cdf.globalattsget().items():
        print(f"Global attribute: {key} = {value}")

    for variable in cdf.cdf_info().zVariables:
        print()
        print(f"Variable: {variable}")
        print(f"Shape: {cdf.varget(variable).shape}")
        print(f"Data type: {cdf.varinq(variable).Data_Type_Description}")
        for key, value in cdf.varattsget(variable).items():
            print(f"Variable attribute: {key} = {value}")

def print_entire(array):
    for i in range(len(array)):
        print(f"Index {i}: {array[i]}")
    return