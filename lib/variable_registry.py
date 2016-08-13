from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from functools import wraps
from collections import Iterable
from tensorflow import Variable as tf_Variable

from six.moves import range, zip

def _vr_flatten(x):
    if isinstance(x, Iterable) and not isinstance(x, tf_Variable):
        # yield from (a for i in x for a in _vr_flatten(i))
        for y in (a for i in x for a in _vr_flatten(i)):
            yield y
    else:
        yield x


class VariableRegistry():
    def __init__(self):
        self.params = []
        self.current_index = -1
        self.check_out_only = False
        self.weight_vars = []
        self.bias_vars = []
        
    def register(self, param, is_weight=True):
        self.params.append(param)
        if is_weight:
            self.weight_vars.append(param)
        else:
            self.bias_vars.append(param)
        
    def check_out(self):
        self.current_index += 1
        return self.params[self.current_index]
    
    def reset_index(self):
        self.current_index = -1
        
    def get_param_items(self):
        # yield from _vr_flatten(self.params)
        for i in _vr_flatten(self.params):
            yield i
        
    def get_weight_items(self):
        # yield from _vr_flatten(self.weight_vars)
        for i in _vr_flatten(self.weight_vars):
            yield i


class RegistryManager():
    def __init__(self):
        self.reg_index = {}
        self.reg_stack = []
    
    def push_registry(self, registry_name):
        self.reg_stack.append(registry_name)
        
    def pop_registry(self):
        return self.reg_stack.pop()
    
    def get_registry(self, registry_name):
        reg = self.reg_index.get(registry_name, None)
        if reg is None:
            reg = VariableRegistry()
            self.reg_index[registry_name] = reg
        return reg
    
    def get_active_registry(self):
        if len(self.reg_stack) == 0:
            raise RuntimeError("No active variable registry." +
                  " Registry stack is empty. Use @memoise_variables decorator.")
        return self.get_registry(self.reg_stack[-1])
    
    def hard_reset(self):
        self.reg_index = {}
        self.reg_stack = []
        
    def get_param_items(self):
        # yield from (i for r in self.reg_index.values() 
        #               for i in r.get_param_items())
        for x in (i for r in self.reg_index.values()
                    for i in r.get_param_items()):
            yield x
        
    def get_weight_items(self):
        # yield from (i for r in self.reg_index.values() 
        #               for i in r.get_weight_items())
        for x in (i for r in self.reg_index.values() 
                    for i in r.get_weight_items()):
            yield x
    

DEFAULT_REGISTRY_MANAGER = RegistryManager()


def registered_variable(is_weight=True, reg_mngr=DEFAULT_REGISTRY_MANAGER):
    def pre_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            var_registry = reg_mngr.get_active_registry()
            if var_registry.check_out_only:
                return var_registry.check_out()
            else:
                param = func(*args, **kwargs)
                var_registry.register(param, is_weight=is_weight)
                return param
        return wrapper
    return pre_wrapper


def memoise_variables(registry_name, reg_mngr=DEFAULT_REGISTRY_MANAGER):
    def pre_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            reg_mngr.push_registry(registry_name)
            var_registry = reg_mngr.get_active_registry()
            
            var_registry.reset_index()
            model = func(*args, **kwargs)
            var_registry.check_out_only = True
            
            reg_mngr.pop_registry()
            return model
        return wrapper  
    return pre_wrapper
