# CHAT GPT

import inspect

def is_pybind11_object(obj):
    t = type(obj)
    
    # Must be a type
    if not isinstance(t, type):
        return False
    
    # Must be a C extension type (not pure Python)
    if not hasattr(t, "__flags__"):
        return False
    
    # Heuristic: pybind11 sets __module__ to the extension module
    mod = getattr(t, "__module__", "")
    if not mod or mod == "builtins":
        return False
    
    # Optional: pybind11 sometimes sets __cpp_name__ on the class
    if hasattr(t, "__cpp_name__"):
        return True
    
    # Fallback: looks like a C extension type
    return True

def check_value(value):
    if isinstance(value, list):
        newValue = []
        for i in range(len(value)):
            newValue.insert(i, check_value(value[i]))

        return newValue
    elif is_pybind11_object(value):
        # print("CONVERT OBJECT INTO DICT")
        return pybind_to_dict(value)
    else:
        return value

def pybind_to_dict(obj):
    result = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
            if inspect.ismethod(value) or inspect.isfunction(value):
                continue
            if not value is None:
                result[name] = check_value(value)
        except Exception:
            pass
    # print(result)
    return result