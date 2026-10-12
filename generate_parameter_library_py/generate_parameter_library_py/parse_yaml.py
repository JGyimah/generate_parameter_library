#!/usr/bin/env python3

# Copyright 2023 PickNik Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the PickNik Inc. nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from jinja2 import Template, Environment
from typeguard import typechecked

# try to import TypeCheckError from typeguard. This was breaking and replaced TypeError in 3.0.0
try:
    from typeguard import TypeCheckError
except ImportError as e:
    # otherwise, use the old TypeError
    TypeCheckError = TypeError

from typing import Any, List, Union
from yaml.parser import ParserError
from yaml.scanner import ScannerError
import os
import re
import yaml

from generate_parameter_library_py.cpp_conversions import CPPConversions
from generate_parameter_library_py.python_conversions import PythonConversions
from generate_parameter_library_py.string_filters_cpp import (
    valid_string_cpp,
    valid_string_python,
)


# YAMLSyntaxError standardizes compiler error messages
class YAMLSyntaxError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


# helper functions
@typechecked
def compile_error(msg: str):
    return YAMLSyntaxError('\nERROR: ' + msg)


@typechecked
def array_type(defined_type: str):
    return defined_type.__contains__('array')


@typechecked
def is_mapped_parameter(param_name: str):
    return param_name.__contains__('__map_')


@typechecked
def fixed_type_size(yaml_type: str):
    tmp = yaml_type.split('_')
    if len(tmp) < 3:
        return None
    if tmp[-2] != 'fixed' or not tmp[-1].isdigit():
        return None
    return int(tmp[-1])


@typechecked
def pascal_case(string: str):
    words = string.split('_')
    return ''.join(w.title() for w in words)


@typechecked
def int_to_integer_str(value: str):
    return value.replace('int', 'integer')


@typechecked
def validation_base_name(function_name: str):
    return function_name.replace('<>', '')


# Reserved YAML key holding cross-parameter constraints for the enclosing struct
CONSTRAINTS_KEY = '__constraints'

# Keys allowed alongside the operator inside a single constraint definition
CONSTRAINT_RESERVED_KEYS = {'description', 'when', 'require'}

# Keys forming the conditional (guarded) constraint form
CONSTRAINT_CONDITIONAL_KEYS = {'when', 'require'}

# Built-in cross-parameter relations. These are emitted as inline comparisons,
# so one operator string serves both C++ and Python.
BUILTIN_CONSTRAINTS = {
    'lt': '<',
    'lt_eq': '<=',
    'gt': '>',
    'gt_eq': '>=',
    'eq': '==',
    'ne': '!=',
}

# Relations that require an ordering, and therefore numeric operands
ORDERING_CONSTRAINTS = {'lt', 'lt_eq', 'gt', 'gt_eq'}

NUMERIC_TYPES = {'int', 'double'}
EQUALITY_TYPES = NUMERIC_TYPES | {'string', 'bool'}

CONSTRAINT_NAME_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')

# the alias the generated python module imports the user validation file under
PYTHON_VALIDATOR_NAMESPACE = 'custom_validators'


@typechecked
def is_parameter_reference(value: Any):
    return isinstance(value, str) and value.startswith('$')


@typechecked
def literal_type(value: Any):
    # bool is checked first because it is a subclass of int in Python
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, int):
        return 'int'
    if isinstance(value, float):
        return 'double'
    if isinstance(value, str):
        return 'string'
    return None


@typechecked
def validate_fixed_default_size(
    param_name: str,
    defined_type: str,
    default_value: Any,
    validations_dict: dict,
):
    capacity = fixed_type_size(defined_type)
    if capacity is None or default_value is None:
        return
    if array_type(defined_type) and not isinstance(default_value, list):
        return
    if not array_type(defined_type) and not isinstance(default_value, str):
        return

    default_size = len(default_value)
    if default_size > capacity:
        raise compile_error(
            'Parameter {} has a default value of size {}, which exceeds '
            "the capacity of type '{}' ({}).".format(
                param_name, default_size, defined_type, capacity
            )
        )

    for function_name, arguments in validations_dict.items():
        if validation_base_name(function_name) != 'fixed_size':
            continue

        expected_size = arguments
        if isinstance(arguments, list) and len(arguments) == 1:
            expected_size = arguments[0]
        if isinstance(expected_size, int) and default_size != expected_size:
            raise compile_error(
                'Parameter {} has a default value of size {}, but its '
                "'fixed_size' validation requires {}.".format(
                    param_name, default_size, expected_size
                )
            )


@typechecked
def validate_validator_combinations(param_name: str, validations_dict: dict):
    validation_names = {validation_base_name(name) for name in validations_dict}

    if 'element_bounds' in validation_names and {
        'lower_element_bounds',
        'upper_element_bounds',
    }.intersection(validation_names):
        raise compile_error(
            "Parameter {} cannot combine 'element_bounds' with 'lower_element_bounds/upper_element_bounds'.".format(
                param_name
            )
        )

    scalar_bound_validators = {'gt', 'gt_eq', 'lt', 'lt_eq'}
    if 'bounds' in validation_names and validation_names.intersection(
        scalar_bound_validators
    ):
        raise compile_error(
            "Parameter {} cannot combine 'bounds' with scalar bound validators "
            "(gt/gt_eq/lt/lt_eq). Use only 'bounds<>' for inclusive ranges, "
            'or only scalar bound validators.'.format(param_name)
        )


def get_dynamic_parameter_field(yaml_parameter_name: str):
    tmp = yaml_parameter_name.split('.')
    num_nested = [i for i, val in enumerate(tmp) if is_mapped_parameter(val)]
    field = tmp[(max(num_nested) + 1) :] if len(num_nested) else tmp[-1]
    return '.'.join(field)


def get_dynamic_mapped_parameter(
    yaml_parameter_name: str, declared_names: set = None
) -> list:
    """Get the resolved paths to the key arrays for each __map_ segment."""
    keys = get_dynamic_mapped_parameter_keys(yaml_parameter_name)
    if not declared_names:
        return keys

    struct_name = get_dynamic_struct_name(yaml_parameter_name)
    return [_resolve_key_path(k, struct_name, declared_names) for k in keys]


def _resolve_key_path(key_name: str, struct_name: str, declared_names: set) -> str:
    """Resolve the full path to a mapped parameter key array by searching scopes."""
    prefix = struct_name
    while prefix:
        candidate = f'{prefix}.{key_name}'
        if candidate in declared_names:
            return candidate
        # Move up one level (e.g., "a.b.c" -> "a.b")
        prefix, _, _ = prefix.rpartition('.')
    # Fallback to the bare key name if not found in any scope
    return key_name


def get_dynamic_mapped_parameter_keys(yaml_parameter_name: str) -> list:
    """Get the base key names (without __map_ prefix) for all mapped segments."""
    segments = yaml_parameter_name.split('.')
    return [
        seg.replace('__map_', '') for seg in segments[:-1] if is_mapped_parameter(seg)
    ]


def get_dynamic_struct_name(yaml_parameter_name: str):
    tmp = yaml_parameter_name.split('.')
    num_nested = [i for i, val in enumerate(tmp) if is_mapped_parameter(val)]
    struct_name = tmp[: (min(num_nested))] if len(num_nested) else ''
    return '.'.join(struct_name)


def get_dynamic_parameter_name(yaml_parameter_name: str):
    struct_name = get_dynamic_struct_name(yaml_parameter_name)
    parameter_field = get_dynamic_parameter_field(yaml_parameter_name)
    parameter_name = [struct_name, parameter_field]
    parameter_name = '.'.join(parameter_name)
    return parameter_name


def get_dynamic_parameter_map(yaml_parameter_name: str):
    mapped_params = get_dynamic_mapped_parameter_keys(yaml_parameter_name)
    parameter_map = [val + '_map' for val in mapped_params]
    parameter_map = '.'.join(parameter_map)
    return parameter_map


@typechecked
def is_fixed_type(yaml_type: str):
    return fixed_type_size(yaml_type) is not None


@typechecked
def get_fixed_base_type(yaml_type: str):
    tmp = yaml_type.split('_')
    if len(tmp) >= 2 and tmp[-1] == 'fixed':
        return '_'.join(tmp[:-1])
    if len(tmp) >= 3 and tmp[-2] == 'fixed' and tmp[-1].isdigit():
        return '_'.join(tmp[:-2])
    return '_'.join(tmp[: -(min(2, len(tmp) - 1))])


@typechecked
def get_fixed_type(yaml_type: str):
    return get_fixed_base_type(yaml_type) + '_fixed'


class CodeGenVariableBase:
    @typechecked
    def __init__(
        self,
        language: str,
        name: str,
        param_name: str,
        defined_type: str,
        default_value: Any,
    ):
        if language == 'cpp':
            self.conversion = CPPConversions()
        elif language == 'rst' or language == 'markdown':
            # cpp is used here because it the desired style of the markdown,
            # e.g. "false" for C++ instead of "False" for Python
            self.conversion = CPPConversions()
        elif language == 'python':
            self.conversion = PythonConversions()
        else:
            raise compile_error(
                'Invalid language, only c++ and python are currently supported.'
            )

        self.name = name
        self.default_value = default_value
        self.name = name
        self.param_name = param_name
        self.defined_type, template = self.process_type(defined_type)
        self.array_type = array_type(self.defined_type)

        if self.defined_type not in self.conversion.defined_type_to_lang_type:
            allowed = ', '.join(
                key for key in self.conversion.defined_type_to_lang_type
            )
            raise compile_error(
                f'Invalid parameter type `{defined_type}` for parameter {param_name}. Allowed types are: '
                + allowed
            )
        func = self.conversion.defined_type_to_lang_type[self.defined_type]
        self.lang_type = func(self.defined_type, template)
        tmp = defined_type.split('_')
        self.defined_base_type = tmp[0]
        func = self.conversion.defined_type_to_lang_type[self.defined_base_type]
        self.lang_base_type = func(self.defined_base_type, template)
        func = self.conversion.lang_str_value_func[self.defined_type]
        try:
            self.lang_str_value = func(default_value)
        except TypeCheckError:
            raise compile_error(
                f'Parameter {param_name} has incorrect type. Expected: {defined_type}, got: {self.get_yaml_type_from_python(default_value)}'
            )

    def parameter_as_function_str(self):
        if self.defined_type not in self.conversion.yaml_type_to_as_function:
            raise compile_error('invalid yaml type: %s' % type(self.defined_type))
        return self.conversion.yaml_type_to_as_function[self.defined_type]

    def get_python_val_to_str_func(self, arg):
        return self.conversion.python_val_to_str_func[str(type(arg))]

    def get_yaml_type_from_python(self, arg):
        if isinstance(arg, list):
            return self.conversion.python_list_to_yaml_type[str(type(arg[0]))]
        else:
            return self.conversion.python_val_to_yaml_type[str(type(arg[0]))]

    def process_type(self, defined_type):
        raise NotImplemented()

    def get_parameter_type(self):
        raise NotImplemented()


class CodeGenVariable(CodeGenVariableBase):
    def process_type(self, defined_type):
        return defined_type, None

    def get_parameter_type(self):
        return int_to_integer_str(self.defined_type).upper()


class CodeGenFixedVariable(CodeGenVariableBase):
    def process_type(self, defined_type):
        size = fixed_type_size(defined_type)
        tmp = defined_type.split('_')
        yaml_base_type = tmp[0]
        func = self.conversion.defined_type_to_lang_type[yaml_base_type]
        lang_base_type = func(yaml_base_type, None)
        defined_type = get_fixed_type(defined_type)
        return defined_type, (lang_base_type, size)

    def get_parameter_type(self):
        return int_to_integer_str(get_fixed_base_type(self.defined_type)).upper()


# Each template has a corresponding class with the str filling in the template with jinja
class VariableDeclaration:
    @typechecked
    def __init__(self, code_gen_variable: CodeGenVariableBase):
        self.code_gen_variable = code_gen_variable

    def __str__(self):
        value = self.code_gen_variable.lang_str_value
        data = {
            'type': self.code_gen_variable.lang_type,
            'name': self.code_gen_variable.name,
            'value': value,
        }

        j2_template = Template(GenerateCode.templates['declare_variable'])
        code = j2_template.render(data, trim_blocks=True)
        return code


class DeclareStruct:
    @typechecked
    def __init__(self, struct_name: str, fields: List[VariableDeclaration]):
        self.struct_name = struct_name
        self.fields = fields
        self.sub_structs = []
        self.struct_instance = ''

    @typechecked
    def add_field(self, field: VariableDeclaration):
        self.fields.append(field)

    def add_sub_struct(self, sub_struct):
        self.sub_structs.append(sub_struct)

    def field_content(self):
        content = ''.join(str(x) for x in self.fields)
        return str(content)

    def sub_struct_content(self):
        content = ''.join(str(x) for x in self.sub_structs)
        return str(content)

    def python_struct_instance(name):
        return (
            ''
            if is_mapped_parameter(name)
            else f'self.{name} = self.__{pascal_case(name)}()'
        )

    def __str__(self):
        sub_struct_str = ''.join(str(x) for x in self.sub_structs)
        field_str = ''.join(str(x) for x in self.fields)
        if field_str == '' and sub_struct_str == '':
            return ''

        # Special case for python: Instance must be added separated to be placed in the __init__ call
        sub_struct_python_instances = '\n'.join(
            DeclareStruct.python_struct_instance(x.struct_name)
            for x in self.sub_structs
        )

        if is_mapped_parameter(self.struct_name):
            map_val_type = pascal_case(self.struct_name)
            map_name = self.struct_name.replace('__map_', '') + '_map'
            map_name = map_name.replace('.', '_')
        else:
            map_val_type = ''
            map_name = ''
            self.struct_instance = self.struct_name

        data = {
            'struct_name': pascal_case(self.struct_name),
            'struct_instance': self.struct_instance,
            'struct_fields': str(field_str),
            'sub_structs': str(sub_struct_str),
            'sub_struct_python_instances': sub_struct_python_instances,
            'map_value_type': map_val_type,
            'map_name': map_name,
        }

        j2_template = Template(GenerateCode.templates['declare_struct'])
        code = j2_template.render(data, trim_blocks=True)
        return code


class ValidationFunction:
    @typechecked
    def __init__(
        self,
        function_name: str,
        arguments: Union[None, List[Any]],
        code_gen_variable: CodeGenVariableBase,
    ):
        self.code_gen_variable = code_gen_variable
        self.function_name = function_name
        self.function_base_name = function_name.replace('<>', '')

        if arguments is not None:
            self.arguments = arguments
        else:
            self.arguments = []

    def __str__(self):
        function_name = self.code_gen_variable.conversion.get_func_signature(
            self.function_name, self.code_gen_variable.lang_base_type
        )
        open_bracket = self.code_gen_variable.conversion.open_bracket
        close_bracket = self.code_gen_variable.conversion.close_bracket

        code = function_name + '(param'
        for arg in self.arguments:
            if isinstance(arg, list):
                code += ', ' + open_bracket
                for a in arg[:-1]:
                    val_func = self.code_gen_variable.get_python_val_to_str_func(a)
                    code += val_func(a) + ', '
                val_func = self.code_gen_variable.get_python_val_to_str_func(arg[-1])
                code += val_func(arg[-1])
                code += close_bracket
            else:
                val_func = self.code_gen_variable.get_python_val_to_str_func(arg)
                code += ', ' + val_func(arg)
        code += ')'

        return code


class ParameterValidation:
    @typechecked
    def __init__(
        self,
        invalid_effect: str,
        valid_effect: str,
        validation_function: ValidationFunction,
    ):
        self.invalid_effect = invalid_effect
        self.valid_effect = valid_effect
        self.validation_function = validation_function

    def __str__(self):
        data = {
            'validation_function': str(self.validation_function),
            'valid_effect': self.valid_effect,
            'invalid_effect': self.invalid_effect,
        }

        j2_template = Template(GenerateCode.templates['parameter_validation'])
        code = j2_template.render(data, trim_blocks=True)
        return code


class ConstraintOperand:
    """One side of a cross-parameter constraint: a parameter reference or a literal."""

    @typechecked
    def __init__(
        self,
        conversion,
        defined_type: str,
        param_path: Union[None, str] = None,
        value: Any = None,
        value_known: bool = True,
    ):
        self.conversion = conversion
        self.defined_type = defined_type
        self.param_path = param_path
        self.value = value
        self.value_known = value_known

    @typechecked
    def is_reference(self) -> bool:
        return self.param_path is not None

    @typechecked
    def code(self) -> str:
        """The expression used inside the generated condition."""
        if self.is_reference():
            return 'updated_params.' + self.param_path
        func = self.conversion.python_val_to_str_func[str(type(self.value))]
        return func(self.value)

    @typechecked
    def describe(self) -> str:
        """Human readable text for the runtime error message."""
        if self.is_reference():
            # the value is substituted at runtime by the format placeholder
            return self.param_path + ' ({})'
        if isinstance(self.value, bool):
            return str(self.value).lower()
        return str(self.value)

    @typechecked
    def format_arg(self) -> Union[None, str]:
        """The argument feeding this operand's placeholder, if it has one."""
        return self.code() if self.is_reference() else None

    @typechecked
    def label(self) -> str:
        """Static text for this operand, with no runtime placeholder."""
        if self.is_reference():
            return self.param_path
        if isinstance(self.value, bool):
            return str(self.value).lower()
        if isinstance(self.value, str):
            return "'{}'".format(self.value)
        return str(self.value)


class ConstraintRelation:
    """A single comparison between two operands."""

    @typechecked
    def __init__(self, operator: str, operands: List[ConstraintOperand]):
        self.operator = operator
        self.operands = operands

    @typechecked
    def code(self) -> str:
        lhs, rhs = self.operands
        return '{} {} {}'.format(
            lhs.code(), BUILTIN_CONSTRAINTS[self.operator], rhs.code()
        )

    @typechecked
    def describe(self) -> str:
        """Text with a format placeholder for every parameter reference."""
        lhs, rhs = self.operands
        return '{} must be {} {}'.format(
            lhs.describe(), BUILTIN_CONSTRAINTS[self.operator], rhs.describe()
        )

    @typechecked
    def label(self) -> str:
        """Static text with no placeholders, used to describe a guard."""
        lhs, rhs = self.operands
        return '{} {} {}'.format(
            lhs.label(), BUILTIN_CONSTRAINTS[self.operator], rhs.label()
        )

    @typechecked
    def format_args(self) -> List[str]:
        return [op.format_arg() for op in self.operands if op.format_arg()]

    @typechecked
    def evaluate(self) -> Union[None, bool]:
        """Fold against declared defaults; None when a value is unknown."""
        if not all(op.value_known and op.value is not None for op in self.operands):
            return None
        lhs, rhs = (op.value for op in self.operands)
        return {
            'lt': lhs < rhs,
            'lt_eq': lhs <= rhs,
            'gt': lhs > rhs,
            'gt_eq': lhs >= rhs,
            'eq': lhs == rhs,
            'ne': lhs != rhs,
        }[self.operator]


class ConstraintCall:
    """A call to a user-supplied function relating several parameters.

    Operands are passed positionally by value, so the function's signature is
    checked by the host language's compiler and the generator knows exactly
    which parameters take part.
    """

    @typechecked
    def __init__(
        self, function_name: str, operands: List[ConstraintOperand], conversion
    ):
        self.function_name = function_name
        self.operands = operands
        self.conversion = conversion

    @typechecked
    def code(self) -> str:
        return '{}({})'.format(
            self.conversion.constraint_function_name(self.function_name),
            ', '.join(op.code() for op in self.operands),
        )

    @typechecked
    def evaluate(self) -> Union[None, bool]:
        # an arbitrary predicate cannot be folded at generation time
        return None


class ParameterConstraint:
    """A constraint relating several parameters, checked once per struct.

    Unconditional constraints hold at all times. A conditional constraint
    carries a guard and is only required to hold while that guard is true.
    """

    @typechecked
    def __init__(
        self,
        name: str,
        require: Union[ConstraintRelation, ConstraintCall],
        description: str,
        conversion,
        invalid_effect_func,
        context: str = '',
        guard: Union[None, ConstraintRelation] = None,
    ):
        self.name = name
        self.require = require
        self.description = description
        self.conversion = conversion
        self.invalid_effect_func = invalid_effect_func
        self.context = context
        self.guard = guard

    @typechecked
    def prefix(self) -> str:
        condition = ''
        if self.guard is not None:
            condition = 'with {}, '.format(self.guard.label())
        return "{}Constraint '{}' violated: {}".format(
            self.context, self.name, condition
        )

    @typechecked
    def message(self) -> str:
        if isinstance(self.require, ConstraintCall):
            # the user function supplies the substance of the message
            return self.conversion.constraint_custom_message(self.prefix())
        text = self.prefix() + self.require.describe()
        return self.conversion.constraint_message(text, self.require.format_args())

    def __str__(self):
        invalid_effect = self.invalid_effect_func(self.message())
        if isinstance(self.require, ConstraintCall):
            check = Template(
                GenerateCode.templates['parameter_constraint_custom']
            ).render({'call': self.require.code(), 'invalid_effect': invalid_effect})
        else:
            check = Template(GenerateCode.templates['parameter_constraint']).render(
                {
                    'condition': self.require.code(),
                    'invalid_effect': invalid_effect,
                }
            )
        if self.guard is None:
            return check
        return Template(
            GenerateCode.templates['parameter_constraint_conditional']
        ).render({'guard': self.guard.code(), 'body': check})


class UpdateParameterBase:
    @typechecked
    def __init__(self, parameter_name: str, code_gen_variable: CodeGenVariableBase):
        self.parameter_name = parameter_name
        self.parameter_as_function = code_gen_variable.parameter_as_function_str()
        self.parameter_validations = []

    @typechecked
    def add_parameter_validation(self, parameter_validation: ParameterValidation):
        self.parameter_validations.append(parameter_validation)


class UpdateParameter(UpdateParameterBase):
    def __str__(self):
        parameter_validations_str = ''.join(str(x) for x in self.parameter_validations)

        data = {
            'parameter_name': self.parameter_name,
            'parameter_validations': str(parameter_validations_str),
            'parameter_as_function': self.parameter_as_function,
        }

        j2_template = Template(GenerateCode.templates['update_parameter'])
        code = j2_template.render(data, trim_blocks=True)
        return code


class UpdateRuntimeParameter(UpdateParameterBase):
    def __init__(
        self,
        parameter_name: str,
        code_gen_variable: CodeGenVariableBase,
        mapped_params: list = None,
    ):
        super().__init__(parameter_name, code_gen_variable)
        self.mapped_params = mapped_params or []

    def __str__(self):
        parameter_validations_str = ''.join(str(x) for x in self.parameter_validations)
        mapped_params = self.mapped_params
        parameter_map = get_dynamic_parameter_map(self.parameter_name)
        parameter_map = parameter_map.split('.')
        struct_name = get_dynamic_struct_name(self.parameter_name)
        parameter_field = get_dynamic_parameter_field(self.parameter_name)

        data = {
            'mapped_params': mapped_params,
            'parameter_map': parameter_map,
            'struct_name': struct_name,
            'parameter_field': parameter_field,
            'parameter_validations': str(parameter_validations_str),
            'parameter_as_function': self.parameter_as_function,
        }

        j2_template = Template(GenerateCode.templates['update_runtime_parameter'])
        code = j2_template.render(data, trim_blocks=True)
        return code


class SetStackParams:
    @typechecked
    def __init__(self, parameter_name: str):
        self.parameter_name = parameter_name

    def __str__(self):
        data = {'parameter_name': self.parameter_name}
        j2_template = Template(GenerateCode.templates['set_stack_params'])
        code = j2_template.render(data, trim_blocks=True)
        return code


class SetParameterBase:
    @typechecked
    def __init__(self, parameter_name: str, code_gen_variable: CodeGenVariableBase):
        self.parameter_name = parameter_name
        self.parameter_as_function = code_gen_variable.parameter_as_function_str()
        self.parameter_validations = []

    @typechecked
    def add_parameter_validation(self, parameter_validation: ParameterValidation):
        self.parameter_validations.append(parameter_validation)


class SetParameter(SetParameterBase):
    def __str__(self):
        parameter_validations_str = ''.join(str(x) for x in self.parameter_validations)

        data = {
            'parameter_name': self.parameter_name,
            'parameter_validations': str(parameter_validations_str),
            'parameter_as_function': self.parameter_as_function,
        }

        j2_template = Template(GenerateCode.templates['set_parameter'])
        code = j2_template.render(data, trim_blocks=True)
        return code


class SetRuntimeParameter(SetParameterBase):
    def __str__(self):
        parameter_validations_str = ''.join(str(x) for x in self.parameter_validations)
        parameter_field = get_dynamic_parameter_field(self.parameter_name)
        data = {
            'parameter_field': parameter_field,
            'parameter_validations': str(parameter_validations_str),
            'parameter_as_function': self.parameter_as_function,
        }

        j2_template = Template(GenerateCode.templates['set_runtime_parameter'])
        code = j2_template.render(data, trim_blocks=True)
        return code


class DeclareParameterBase:
    @typechecked
    def __init__(
        self,
        code_gen_variable: CodeGenVariableBase,
        parameter_description: str,
        parameter_read_only: bool,
        parameter_validations: list,
        parameter_additional_constraints: str,
    ):
        self.parameter_name = code_gen_variable.param_name
        self.parameter_description = parameter_description
        self.parameter_read_only = parameter_read_only
        self.parameter_validations = parameter_validations
        self.code_gen_variable = code_gen_variable
        self.parameter_additional_constraints = parameter_additional_constraints


class DeclareParameter(DeclareParameterBase):
    def __str__(self):
        if len(self.code_gen_variable.lang_str_value) == 0:
            self.parameter_value = ''
        else:
            self.parameter_value = self.parameter_name
        bool_to_str = self.code_gen_variable.conversion.bool_to_str

        parameter_validations = self.parameter_validations

        data = {
            'parameter_name': self.parameter_name,
            'parameter_value': self.parameter_value,
            'parameter_type': self.code_gen_variable.get_parameter_type(),
            'parameter_description': self.parameter_description,
            'parameter_read_only': bool_to_str(self.parameter_read_only),
            'parameter_additional_constraints': self.parameter_additional_constraints,
            'parameter_validations': parameter_validations,
        }

        # Create a Jinja2 environment to register the custom filter
        env = Environment()
        env.filters['valid_string_cpp'] = valid_string_cpp
        env.filters['valid_string_python'] = valid_string_python
        j2_template = env.from_string(GenerateCode.templates['declare_parameter'])
        code = j2_template.render(data, trim_blocks=True)
        return code


class DeclareRuntimeParameter(DeclareParameterBase):
    def __init__(
        self,
        code_gen_variable: CodeGenVariableBase,
        parameter_description: str,
        parameter_read_only: bool,
        parameter_validations: list,
        parameter_additional_constraints: str,
        mapped_params: list = None,
    ):
        super().__init__(
            code_gen_variable,
            parameter_description,
            parameter_read_only,
            parameter_validations,
            parameter_additional_constraints,
        )
        self.mapped_params = mapped_params or []
        self.set_runtime_parameter = None
        self.param_struct_instance = 'updated_params'

    @typechecked
    def add_set_runtime_parameter(self, set_runtime_parameter: SetRuntimeParameter):
        self.set_runtime_parameter = set_runtime_parameter

    def __str__(self):
        if self.set_runtime_parameter is None:
            raise AssertionError(
                'add_set_runtime_parameter was not called before str()'
            )

        if self.code_gen_variable.default_value is None:
            default_value = ''
        else:
            default_value = 'non-empty'

        bool_to_str = self.code_gen_variable.conversion.bool_to_str
        parameter_field = get_dynamic_parameter_field(self.parameter_name)
        mapped_params = self.mapped_params
        parameter_map = get_dynamic_parameter_map(self.parameter_name)
        struct_name = get_dynamic_struct_name(self.parameter_name)
        parameter_map = parameter_map.split('.')

        data = {
            'struct_name': struct_name,
            'parameter_type': self.code_gen_variable.get_parameter_type(),
            'parameter_description': self.parameter_description,
            'parameter_read_only': bool_to_str(self.parameter_read_only),
            'parameter_as_function': self.code_gen_variable.parameter_as_function_str(),
            'parameter_additional_constraints': self.parameter_additional_constraints,
            'mapped_params': mapped_params,
            'mapped_param_underscore': [val.replace('.', '_') for val in mapped_params],
            'set_runtime_parameter': self.set_runtime_parameter,
            'parameter_map': parameter_map,
            'param_struct_instance': self.param_struct_instance,
            'parameter_field': parameter_field,
            'default_value': default_value,
            'parameter_validations': self.parameter_validations,
        }

        # Create a Jinja2 environment to register the custom filter
        env = Environment()
        env.filters['valid_string_cpp'] = valid_string_cpp
        env.filters['valid_string_python'] = valid_string_python
        j2_template = env.from_string(
            GenerateCode.templates['declare_runtime_parameter']
        )
        code = j2_template.render(data, trim_blocks=True)
        return code


class RemoveRuntimeParameter:
    @typechecked
    def __init__(self, dynamic_declare_parameter: DeclareRuntimeParameter):
        self.dynamic_declare_parameter = dynamic_declare_parameter

    def __str__(self):
        parameter_map = get_dynamic_parameter_map(
            self.dynamic_declare_parameter.parameter_name
        )
        parameter_map = parameter_map.split('.')
        struct_name = get_dynamic_struct_name(
            self.dynamic_declare_parameter.parameter_name
        )
        parameter_field = get_dynamic_parameter_field(
            self.dynamic_declare_parameter.parameter_name
        )
        mapped_params = self.dynamic_declare_parameter.mapped_params

        data = {
            'parameter_map': parameter_map,
            'mapped_params': mapped_params,
            'dynamic_declare_parameter': str(self.dynamic_declare_parameter),
            'struct_name': struct_name,
            'parameter_field': parameter_field,
        }

        j2_template = Template(GenerateCode.templates['remove_runtime_parameter'])
        code = j2_template.render(data, trim_blocks=True)
        return code


def get_all_templates(language: str):
    template_lang_path = os.path.join(
        os.path.dirname(__file__), 'jinja_templates', language
    )
    if language == 'markdown':
        template_markdown_path = os.path.join(
            os.path.dirname(__file__), 'jinja_templates', 'markdown'
        )
        template_paths = [template_lang_path, template_markdown_path]
    elif language == 'rst':
        template_rst_path = os.path.join(
            os.path.dirname(__file__), 'jinja_templates', 'rst'
        )
        template_paths = [template_lang_path, template_rst_path]
    else:
        template_paths = [template_lang_path]

    template_map = {}
    for template_path in template_paths:
        for file_name in [
            f
            for f in os.listdir(template_path)
            if os.path.isfile(os.path.join(template_path, f))
        ]:
            with open(os.path.join(template_path, file_name)) as file:
                template_map[file_name] = file.read()

    return template_map


def preprocess_inputs(language, name, value, nested_name_list):
    # define parameter name
    param_name = ''.join(x + '.' for x in nested_name_list[1:]) + name

    # required attributes
    try:
        defined_type = value['type']
    except KeyError as e:
        raise compile_error('No type defined for parameter %s' % param_name)

    # check for invalid syntax
    valid_keys = {
        'default_value',
        'description',
        'read_only',
        'additional_constraints',
        'validation',
        'type',
    }
    invalid_keys = value.keys() - valid_keys
    if len(invalid_keys) > 0:
        raise compile_error(
            "Invalid syntax in parameter %s. '%s' is not valid syntax"
            % (param_name, next(iter(invalid_keys)))
        )

    # optional attributes
    default_value = value.get('default_value', None)
    validations_dict = value.get('validation', {})
    validate_fixed_default_size(
        param_name, defined_type, default_value, validations_dict
    )
    if not is_fixed_type(defined_type):
        code_gen_variable = CodeGenVariable(
            language, name, param_name, defined_type, default_value
        )
    else:
        code_gen_variable = CodeGenFixedVariable(
            language, name, param_name, defined_type, default_value
        )

    description = value.get('description', '')
    read_only = bool(value.get('read_only', False))
    validations = []
    additional_constraints = value.get('additional_constraints', '')
    if is_fixed_type(defined_type):
        validations_dict['size_lt<>'] = fixed_type_size(defined_type) + 1

    validate_validator_combinations(param_name, validations_dict)

    for func_name in validations_dict:
        args = validations_dict[func_name]
        if args is not None and not isinstance(args, list):
            args = [args]
        validations.append(ValidationFunction(func_name, args, code_gen_variable))

    return (
        code_gen_variable,
        description,
        read_only,
        validations,
        additional_constraints,
    )


# class used to generate c++ code from yaml file
class GenerateCode:
    templates = None

    def __init__(self, language: str):
        if language == 'cpp':
            self.comments = '// auto-generated DO NOT EDIT'
        elif language == 'rst':
            self.comments = '.. auto-generated DO NOT EDIT'
        elif language == 'markdown':
            self.comments = '<!--- auto-generated DO NOT EDIT -->'
        elif language == 'python' or language == 'markdown':
            self.comments = '# auto-generated DO NOT EDIT'
        else:
            raise compile_error(
                'Invalid language, only cpp, markdown, rst, and python are currently supported.'
            )
        GenerateCode.templates = get_all_templates(language)
        self.declared_param_names = set()
        self.language = language
        self.namespace = ''
        self.struct_tree = DeclareStruct('Params', [])
        self.stack_struct_tree = DeclareStruct('StackParams', [])
        self.update_parameters = []
        self.declare_parameters = []
        self.declare_dynamic_parameters = []
        self.update_dynamic_parameters = []
        self.update_declare_dynamic_parameter = []
        self.remove_dynamic_parameter = []
        self.declare_parameter_sets = []
        self.set_stack_params = []
        self.user_validation_file = ''
        # cross-parameter constraints
        if language == 'python':
            self.conversion = PythonConversions()
        else:
            self.conversion = CPPConversions()
        self.declared_param_info = {}
        self.mapped_param_names = set()
        self.collected_constraints = []
        self.update_constraints = []
        self.declare_constraints = []
        self.custom_constraints_used = False

    def parse(self, yaml_file, validate_header):
        with open(yaml_file) as f:
            try:
                docs = yaml.load_all(f, Loader=yaml.Loader)
                doc = list(docs)[0]
            except ParserError as e:
                raise compile_error(str(e))
            except ScannerError as e:
                raise compile_error(str(e))

            if len(doc) != 1:
                raise compile_error(
                    'The yaml definition must only have one root element'
                )
            self.namespace = list(doc.keys())[0]
            self.user_validation_file = validate_header
            self.parse_dict(self.namespace, doc[self.namespace], [])
            # constraints are resolved in a second pass, because a constraint may
            # reference a parameter declared later in the file
            self.resolve_constraints()

    def parse_params(self, name, value, nested_name_list):
        (
            code_gen_variable,
            description,
            read_only,
            validations,
            additional_constraints,
        ) = preprocess_inputs(self.language, name, value, nested_name_list)
        # Skip accepted params that do not generate code for code-generation targets.
        # Documentation targets still need to keep these entries (e.g. type: none).
        if code_gen_variable.lang_type is None and self.language not in {
            'markdown',
            'rst',
        }:
            return

        param_name = code_gen_variable.param_name
        update_parameter_invalid = (
            code_gen_variable.conversion.update_parameter_fail_validation()
        )
        update_parameter_valid = (
            code_gen_variable.conversion.update_parameter_pass_validation()
        )
        declare_parameter_invalid = (
            code_gen_variable.conversion.initialization_fail_validation(param_name)
        )
        declare_parameter_valid = (
            code_gen_variable.conversion.initialization_pass_validation(param_name)
        )

        # add variable to struct
        var = VariableDeclaration(code_gen_variable)

        # check if runtime parameter
        is_runtime_parameter = is_mapped_parameter(param_name)

        # Resolve mapped paths for dynamic params; track paths for standard params
        mapped_params = []
        if is_runtime_parameter:
            self.mapped_param_names.add(param_name)
            mapped_params = get_dynamic_mapped_parameter(
                param_name, self.declared_param_names
            )
        else:
            self.declared_param_names.add(param_name)
            one_of_values = None
            for validation in validations:
                if validation.function_base_name == 'one_of' and validation.arguments:
                    candidate = validation.arguments[0]
                    if isinstance(candidate, list):
                        one_of_values = candidate
            self.declared_param_info[param_name] = {
                'type': code_gen_variable.defined_type,
                'default': code_gen_variable.default_value,
                'one_of': one_of_values,
            }

        if is_runtime_parameter:
            declare_parameter_set = SetRuntimeParameter(param_name, code_gen_variable)
            declare_parameter = DeclareRuntimeParameter(
                code_gen_variable,
                description,
                read_only,
                validations,
                additional_constraints,
                mapped_params,
            )
            declare_parameter.add_set_runtime_parameter(declare_parameter_set)
            update_parameter = UpdateRuntimeParameter(
                param_name, code_gen_variable, mapped_params
            )
        else:
            declare_parameter = DeclareParameter(
                code_gen_variable,
                description,
                read_only,
                validations,
                additional_constraints,
            )
            declare_parameter_set = SetParameter(param_name, code_gen_variable)
            update_parameter = UpdateParameter(param_name, code_gen_variable)

        # set parameter
        for validation_function in validations:
            parameter_validation = ParameterValidation(
                declare_parameter_invalid, declare_parameter_valid, validation_function
            )
            declare_parameter_set.add_parameter_validation(parameter_validation)

        # update parameter
        for validation_function in validations:
            parameter_validation = ParameterValidation(
                update_parameter_invalid, update_parameter_valid, validation_function
            )
            update_parameter.add_parameter_validation(parameter_validation)

        self.struct_tree.add_field(var)
        if not is_runtime_parameter and (
            isinstance(code_gen_variable, CodeGenFixedVariable)
            or not (
                code_gen_variable.array_type
                or code_gen_variable.defined_type == 'string'
            )
        ):
            self.stack_struct_tree.add_field(var)
            self.set_stack_params.append(SetStackParams(code_gen_variable.param_name))
        if is_runtime_parameter:
            self.declare_dynamic_parameters.append(declare_parameter)
            self.update_dynamic_parameters.append(update_parameter)
            self.update_declare_dynamic_parameter.append(declare_parameter)
            dynamic_update_parameter = RemoveRuntimeParameter(declare_parameter)
            self.remove_dynamic_parameter.append(dynamic_update_parameter)
        else:
            self.declare_parameters.append(declare_parameter)
            self.update_parameters.append(update_parameter)
            self.declare_parameter_sets.append(declare_parameter_set)

    @typechecked
    def collect_constraints(self, scope: str, block: Any):
        """Validate the shape of a __constraints block and stash it for the second pass."""
        location = scope[:-1] if scope else self.namespace
        if not isinstance(block, dict):
            raise compile_error(
                "'{}' in {} must be a map of constraint names to definitions.".format(
                    CONSTRAINTS_KEY, location
                )
            )

        seen = {name for _, name, _ in self.collected_constraints}
        for constraint_name, spec in block.items():
            if not isinstance(
                constraint_name, str
            ) or not CONSTRAINT_NAME_PATTERN.match(constraint_name):
                raise compile_error(
                    "Invalid constraint name '{}' in {}. Constraint names must be "
                    'snake_case and start with a letter.'.format(
                        constraint_name, location
                    )
                )
            if constraint_name in seen:
                raise compile_error(
                    "Duplicate constraint name '{}'. Constraint names must be unique.".format(
                        constraint_name
                    )
                )
            if not isinstance(spec, dict):
                raise compile_error(
                    "Constraint '{}' must be a map containing an operator, "
                    "e.g. 'lt: [$a, $b]'.".format(constraint_name)
                )
            seen.add(constraint_name)
            self.collected_constraints.append((scope, constraint_name, spec))

    @typechecked
    def resolve_reference(self, reference: str, scope: str, constraint_name: str):
        """Resolve a $reference to a declared parameter path."""
        path = reference[1:]
        if not path:
            raise compile_error(
                "Constraint '{}' has an empty parameter reference '$'.".format(
                    constraint_name
                )
            )

        # relative to the struct holding the constraint first, then from the root
        candidates = [scope + path, path] if scope else [path]
        for candidate in candidates:
            if candidate in self.declared_param_info:
                return candidate

        for mapped in self.mapped_param_names:
            if mapped.replace('__map_', '').endswith(path):
                raise compile_error(
                    "Constraint '{}' references '{}', which is inside a mapped "
                    '(__map_) parameter. Constraints cannot reference mapped '
                    'parameters.'.format(constraint_name, path)
                )

        raise compile_error(
            "Constraint '{}' references unknown parameter '{}'. Tried: {}.".format(
                constraint_name, path, ', '.join(candidates)
            )
        )

    @typechecked
    def build_operand(
        self,
        raw: Any,
        scope: str,
        constraint_name: str,
        operator: str,
        custom: bool = False,
    ):
        if is_parameter_reference(raw):
            path = self.resolve_reference(raw, scope, constraint_name)
            info = self.declared_param_info[path]
            if not custom:
                self.check_operand_type(info['type'], constraint_name, operator, path)
            return ConstraintOperand(
                self.conversion,
                info['type'],
                param_path=path,
                value=info['default'],
                value_known=info['default'] is not None,
            )

        defined_type = literal_type(raw)
        if defined_type is None:
            raise compile_error(
                "Constraint '{}' has an unsupported operand '{}'. Operands must be "
                'a $parameter reference or a scalar literal.'.format(
                    constraint_name, raw
                )
            )
        if not custom:
            self.check_operand_type(defined_type, constraint_name, operator, repr(raw))
        return ConstraintOperand(self.conversion, defined_type, value=raw)

    @typechecked
    def check_operand_type(
        self, defined_type: str, constraint_name: str, operator: str, what: str
    ):
        if array_type(defined_type) or is_fixed_type(defined_type):
            raise compile_error(
                "Constraint '{}' cannot use '{}': constraints only support scalar "
                'operands, but it has type {}.'.format(
                    constraint_name, what, defined_type
                )
            )
        allowed = NUMERIC_TYPES if operator in ORDERING_CONSTRAINTS else EQUALITY_TYPES
        if defined_type not in allowed:
            raise compile_error(
                "Constraint '{}' cannot apply '{}' to '{}' of type {}. Allowed "
                'types are: {}.'.format(
                    constraint_name,
                    operator,
                    what,
                    defined_type,
                    ', '.join(sorted(allowed)),
                )
            )

    @typechecked
    def parse_relation(
        self,
        spec: Any,
        scope: str,
        constraint_name: str,
        what: str,
        allow_custom: bool = True,
    ) -> Union[ConstraintRelation, ConstraintCall]:
        """Parse a single ``{operator: [operand, operand]}`` mapping."""
        if not isinstance(spec, dict):
            raise compile_error(
                "Constraint '{}' has an invalid '{}': expected a mapping such as "
                "'lt: [$a, $b]'.".format(constraint_name, what)
            )
        operators = [k for k in spec if k not in CONSTRAINT_RESERVED_KEYS]
        if len(operators) != 1:
            raise compile_error(
                "Constraint '{}' must contain exactly one operator in '{}', "
                'found {}.'.format(constraint_name, what, len(operators))
            )
        operator = operators[0]
        if isinstance(operator, str) and operator.endswith('<>'):
            raise compile_error(
                "Constraint '{}' uses '{}'. The '<>' suffix is only for "
                'per-parameter validators; constraints are not templated.'.format(
                    constraint_name, operator
                )
            )
        if operator not in BUILTIN_CONSTRAINTS:
            if isinstance(operator, str) and '::' in operator:
                if not allow_custom:
                    raise compile_error(
                        "Constraint '{}' uses the custom function '{}' in '{}'. "
                        'Custom functions return a validation result rather than a '
                        'boolean, so they can only be used as the requirement.'.format(
                            constraint_name, operator, what
                        )
                    )
                return self.parse_custom_call(
                    operator, spec[operator], scope, constraint_name
                )
            raise compile_error(
                "Constraint '{}' uses unknown operator '{}' in '{}'. Supported "
                'operators are: {}. A custom function must be fully qualified, '
                "e.g. 'my_project::my_check'.".format(
                    constraint_name,
                    operator,
                    what,
                    ', '.join(sorted(BUILTIN_CONSTRAINTS)),
                )
            )

        raw_operands = spec[operator]
        if not isinstance(raw_operands, list) or len(raw_operands) != 2:
            raise compile_error(
                "Constraint '{}' with operator '{}' takes exactly 2 operands, "
                'e.g. [$vel_min, $vel_max].'.format(constraint_name, operator)
            )

        operands = [
            self.build_operand(raw, scope, constraint_name, operator)
            for raw in raw_operands
        ]
        if not any(op.is_reference() for op in operands):
            raise compile_error(
                "Constraint '{}' has a '{}' that does not reference any parameter. "
                'At least one operand must be a $reference.'.format(
                    constraint_name, what
                )
            )
        return ConstraintRelation(operator, operands)

    @typechecked
    def parse_custom_call(
        self, function_name: str, raw_operands: Any, scope: str, constraint_name: str
    ) -> ConstraintCall:
        if not isinstance(raw_operands, list) or not raw_operands:
            raise compile_error(
                "Constraint '{}' calls '{}' with no operands. A custom constraint "
                'takes a list of operands, at least one of which is a '
                '$reference.'.format(constraint_name, function_name)
            )
        operands = [
            self.build_operand(raw, scope, constraint_name, function_name, custom=True)
            for raw in raw_operands
        ]
        if not any(op.is_reference() for op in operands):
            raise compile_error(
                "Constraint '{}' calls '{}' without referencing any parameter. "
                'At least one operand must be a $reference.'.format(
                    constraint_name, function_name
                )
            )
        # the python template imports the validation module under a fixed alias,
        # so any other namespace would be a NameError at runtime
        if self.language == 'python' and not function_name.startswith(
            PYTHON_VALIDATOR_NAMESPACE + '::'
        ):
            raise compile_error(
                "Constraint '{}' calls '{}'. For python targets a custom function "
                "must be qualified with '{}::', which is the alias the generated "
                'module imports the validation file under.'.format(
                    constraint_name, function_name, PYTHON_VALIDATOR_NAMESPACE
                )
            )
        self.custom_constraints_used = True
        return ConstraintCall(function_name, operands, self.conversion)

    @typechecked
    def parse_guard(
        self, spec: Any, scope: str, constraint_name: str
    ) -> ConstraintRelation:
        """Parse a 'when' guard, accepting the ``{$param: value}`` shorthand."""
        if isinstance(spec, dict) and len(spec) == 1:
            key = next(iter(spec))
            if is_parameter_reference(key):
                spec = {'eq': [key, spec[key]]}
        guard = self.parse_relation(
            spec, scope, constraint_name, 'when', allow_custom=False
        )
        self.check_guard_reachable(guard, constraint_name)
        return guard

    @typechecked
    def check_guard_reachable(
        self, guard: ConstraintRelation, constraint_name: str
    ) -> None:
        """Reject a guard that a one_of validator makes impossible."""
        if guard.operator != 'eq':
            return
        references = [op for op in guard.operands if op.is_reference()]
        literals = [op for op in guard.operands if not op.is_reference()]
        if len(references) != 1 or len(literals) != 1:
            return
        allowed = self.declared_param_info[references[0].param_path]['one_of']
        if allowed is None or literals[0].value in allowed:
            return
        raise compile_error(
            "Constraint '{}' has a 'when' guard that can never be true: "
            "'{}' is not one of the values allowed for {} ({}).".format(
                constraint_name,
                literals[0].value,
                references[0].param_path,
                ', '.join(str(v) for v in allowed),
            )
        )

    @typechecked
    def fold_constraint_defaults(
        self,
        constraint_name: str,
        require: Union[ConstraintRelation, ConstraintCall],
        guard: Union[None, ConstraintRelation],
    ) -> None:
        """Reject a constraint that the declared defaults already violate.

        A guarded constraint is only folded when the guard is provably active
        at the defaults; an inactive guard makes the requirement vacuous.
        """
        if guard is not None and guard.evaluate() is not True:
            return
        if require.evaluate() is not False:
            return
        lhs, rhs = (op.value for op in require.operands)
        condition = ''
        if guard is not None:
            condition = 'with {}, '.format(guard.label())
        raise compile_error(
            "Constraint '{}' is violated by the declared default values: "
            '{}{} {} {} is false.'.format(
                constraint_name,
                condition,
                lhs,
                BUILTIN_CONSTRAINTS[require.operator],
                rhs,
            )
        )

    @typechecked
    def resolve_constraints(self) -> None:
        for scope, constraint_name, spec in self.collected_constraints:
            description = spec.get('description', '')
            conditional_keys = CONSTRAINT_CONDITIONAL_KEYS.intersection(spec)
            operators = [k for k in spec if k not in CONSTRAINT_RESERVED_KEYS]

            if conditional_keys:
                if operators:
                    raise compile_error(
                        "Constraint '{}' mixes '{}' with the conditional "
                        "'when'/'require' form. Use one or the other.".format(
                            constraint_name, operators[0]
                        )
                    )
                missing = CONSTRAINT_CONDITIONAL_KEYS - conditional_keys
                if missing:
                    raise compile_error(
                        "Constraint '{}' has '{}' without '{}'. The conditional "
                        'form needs both.'.format(
                            constraint_name,
                            next(iter(conditional_keys)),
                            next(iter(missing)),
                        )
                    )
                guard = self.parse_guard(spec['when'], scope, constraint_name)
                require = self.parse_relation(
                    spec['require'], scope, constraint_name, 'require'
                )
            else:
                guard = None
                require = self.parse_relation(
                    spec, scope, constraint_name, CONSTRAINTS_KEY
                )

            self.fold_constraint_defaults(constraint_name, require, guard)

            # markdown and rst reuse this parser for documentation only
            if self.language not in {'cpp', 'python'}:
                continue

            self.update_constraints.append(
                ParameterConstraint(
                    constraint_name,
                    require,
                    description,
                    self.conversion,
                    self.conversion.constraint_fail_update,
                    guard=guard,
                )
            )
            self.declare_constraints.append(
                ParameterConstraint(
                    constraint_name,
                    require,
                    description,
                    self.conversion,
                    self.conversion.constraint_fail_init,
                    'Invalid values set during initialization: ',
                    guard=guard,
                )
            )

        # a custom constraint calls into the user's validation header, so the
        # generated code cannot compile without it
        if (
            self.custom_constraints_used
            and self.language in {'cpp', 'python'}
            and not self.user_validation_file
        ):
            raise compile_error(
                'A constraint calls a custom function, but no validation header '
                'was given to the generator. Pass one as the third argument to '
                'generate_parameter_library().'
            )

    def parse_dict(self, name, root_map, nested_name):
        if isinstance(root_map, dict) and isinstance(
            next(iter(root_map.values())), dict
        ):
            cur_struct_tree = self.struct_tree
            cur_stack_struct_tree = self.stack_struct_tree

            sub_struct = DeclareStruct(name, [])
            sub_stack_struct = DeclareStruct(name, [])
            self.struct_tree.add_sub_struct(sub_struct)
            self.struct_tree = sub_struct
            self.stack_struct_tree.add_sub_struct(sub_stack_struct)
            self.stack_struct_tree = sub_stack_struct
            for key in root_map:
                if key == CONSTRAINTS_KEY:
                    scope = ''.join(x + '.' for x in (nested_name + [name])[1:])
                    self.collect_constraints(scope, root_map[key])
                    continue
                if isinstance(root_map[key], dict):
                    nested_name.append(name)
                    self.parse_dict(key, root_map[key], nested_name)
                    nested_name.pop()

            self.struct_tree = cur_struct_tree
            self.stack_struct_tree = cur_stack_struct_tree
        else:
            self.parse_params(name, root_map, nested_name)

    def __str__(self):
        data = {
            'user_validation_file': self.user_validation_file,
            'comments': self.comments,
            'namespace': self.namespace,
            'field_content': self.struct_tree.sub_structs[0].field_content(),
            'sub_struct_content': self.struct_tree.sub_structs[0].sub_struct_content(),
            'sub_struct_python_instances': '\n'.join(
                [
                    DeclareStruct.python_struct_instance(x.struct_name)
                    for x in self.struct_tree.sub_structs[0].sub_structs
                ]
            ),
            'stack_field_content': self.stack_struct_tree.sub_structs[
                0
            ].field_content(),
            'stack_sub_struct_content': self.stack_struct_tree.sub_structs[
                0
            ].sub_struct_content(),
            'update_params_set': '\n'.join([str(x) for x in self.update_parameters]),
            'update_dynamic_parameters': '\n'.join(
                [str(x) for x in self.update_dynamic_parameters]
            ),
            'declare_params': '\n'.join([str(x) for x in self.declare_parameters]),
            'declare_params_set': '\n'.join(
                [str(x) for x in self.declare_parameter_sets]
            ),
            'declare_set_dynamic_params': '\n'.join(
                [str(x) for x in self.declare_dynamic_parameters]
            ),
            'update_declare_dynamic_parameters': '\n'.join(
                [str(x) for x in self.update_declare_dynamic_parameter]
            ),
            'set_stack_params': '\n'.join([str(x) for x in self.set_stack_params]),
            'update_constraints': '\n'.join([str(x) for x in self.update_constraints]),
            'declare_constraints': '\n'.join(
                [str(x) for x in self.declare_constraints]
            ),
            # TODO support removing runtime parameters
            # "remove_dynamic_parameters": "\n".join(
            #     [str(x) for x in self.remove_dynamic_parameter]
            # ),
        }

        j2_template = Template(
            GenerateCode.templates['parameter_library_header'],
            keep_trailing_newline=True,
        )
        code = j2_template.render(data, trim_blocks=True)
        return code
