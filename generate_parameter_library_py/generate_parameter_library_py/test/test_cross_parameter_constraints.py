# Copyright 2026 PickNik Inc.
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

import ast
import os
import tempfile

import pytest

from generate_parameter_library_py.generate_cpp_header import run as run_cpp
from generate_parameter_library_py.generate_python_module import run as run_python
from generate_parameter_library_py.parse_yaml import GenerateCode, YAMLSyntaxError

# Parameters shared by every case below. vel_min/vel_max are scalars with
# defaults that satisfy the constraints, joints is an array, weight lives inside
# a map, and pid.rate has no default.
PARAMETERS = """to_base:
  navigation_mode:
    type: string
    default_value: "{nav_mode}"
    validation:
      one_of<>: [["autonomous", "manual"]]
  vel_min:
    type: double
    default_value: {vel_min}
  vel_max:
    type: double
    default_value: 1.6
  joints:
    type: string_array
    default_value: ["joint1"]
  pid:
    rate:
      type: double
  __map_joints:
    weight:
      type: double
      default_value: 1.0
  __constraints:
{constraints}
"""


def write_yaml(constraints, vel_min='0.5', nav_mode='autonomous'):
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False
    ) as yaml_file:
        yaml_file.write(
            PARAMETERS.format(
                constraints=constraints, vel_min=vel_min, nav_mode=nav_mode
            )
        )
        return yaml_file.name


def generate(language, constraints, vel_min='0.5', header=''):
    yaml_file = write_yaml(constraints, vel_min)
    suffix = '.hpp' if language == 'cpp' else '.py'
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as out:
        output_file = out.name
    try:
        if language == 'cpp':
            run_cpp(output_file, yaml_file, header)
        else:
            run_python(output_file, yaml_file, header)
        with open(output_file) as f:
            return f.read()
    finally:
        os.unlink(yaml_file)
        os.unlink(output_file)


def parse_only(constraints, vel_min='0.5', nav_mode='autonomous', header=''):
    yaml_file = write_yaml(constraints, vel_min, nav_mode)
    try:
        GenerateCode('cpp').parse(yaml_file, header)
    finally:
        os.unlink(yaml_file)


ORDERING = """    velocity_ordering:
      lt: [$vel_min, $vel_max]
      description: "vel_min must be strictly less than vel_max"
"""


def test_cpp_constraint_is_checked_before_params_are_committed():
    code = generate('cpp', ORDERING)
    condition = 'if (!(updated_params.vel_min < updated_params.vel_max))'
    assert code.count(condition) == 2, 'expected a check in update and declare'

    # the update check must sit inside the any_param_updated branch, ahead of
    # the commit, so a rejected set leaves params_ untouched
    update_body = code.split('if (any_param_updated) {')[1]
    assert update_body.index(condition) < update_body.index(
        'update_internal_params(updated_params)'
    )
    assert 'return rsl::to_parameter_result_msg(tl::make_unexpected' in update_body

    # the declare check throws, matching the existing initialization behaviour
    assert 'throw rclcpp::exceptions::InvalidParameterValueException' in code
    assert "Constraint 'velocity_ordering' violated" in code


def test_python_constraint_generates_valid_module():
    code = generate('python', ORDERING)
    ast.parse(code)
    assert code.count('if not (updated_params.vel_min < updated_params.vel_max):') == 2
    assert 'return SetParametersResult(successful=False' in code
    assert 'raise InvalidParameterValueException' in code


def test_literal_operand_is_inlined():
    code = generate('cpp', '    ceiling:\n      lt_eq: [$vel_max, 3.0]\n')
    assert 'if (!(updated_params.vel_max <= 3.0))' in code


def test_reference_resolves_from_enclosing_scope_then_root():
    # pid.rate has no default, so this also covers "unknown value, no folding"
    parse_only('    c:\n      lt: [$pid.rate, $vel_max]\n')


def test_schema_without_constraints_is_unaffected():
    yaml_without = PARAMETERS.format(
        constraints='', vel_min='0.5', nav_mode='autonomous'
    ).replace('  __constraints:\n', '')
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False
    ) as yaml_file:
        yaml_file.write(yaml_without)
        path = yaml_file.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.hpp', delete=False) as out:
        output_file = out.name
    try:
        run_cpp(output_file, path)
        with open(output_file) as f:
            code = f.read()
        assert 'Constraint' not in code
    finally:
        os.unlink(path)
        os.unlink(output_file)


@pytest.mark.parametrize(
    ('constraints', 'vel_min', 'expected'),
    [
        (
            '    bad:\n      lt: [$vel_max, $vel_min]\n',
            '1.6',
            'violated by the declared default values',
        ),
        (
            '    bad:\n      lt: [$vel_min, $does_not_exist]\n',
            '0.5',
            'references unknown parameter',
        ),
        (
            '    bad:\n      lt: [$vel_min, $weight]\n',
            '0.5',
            'inside a mapped (__map_) parameter',
        ),
        (
            '    bad:\n      lt: [$vel_min, $joints]\n',
            '0.5',
            'only support scalar operands',
        ),
        (
            '    bad:\n      lt: [$navigation_mode, "manual"]\n',
            '0.5',
            "cannot apply 'lt'",
        ),
        (
            '    bad:\n      lt<>: [$vel_min, $vel_max]\n',
            '0.5',
            "'<>' suffix is only for per-parameter validators",
        ),
        (
            '    bad:\n      between: [$vel_min, $vel_max]\n',
            '0.5',
            'unknown operator',
        ),
        (
            '    bad:\n      lt: [$vel_min]\n',
            '0.5',
            'takes exactly 2 operands',
        ),
        (
            '    bad:\n      lt: [$vel_min, $vel_max]\n      gt: [$vel_min, $vel_max]\n',
            '0.5',
            'exactly one operator',
        ),
        (
            '    bad:\n      lt: [1.0, 2.0]\n',
            '0.5',
            'does not reference any parameter',
        ),
        (
            '    BadName:\n      lt: [$vel_min, $vel_max]\n',
            '0.5',
            'must be snake_case',
        ),
        (
            '    bad: [1, 2]\n',
            '0.5',
            'must be a map containing an operator',
        ),
    ],
)
def test_invalid_constraints_are_rejected(constraints, vel_min, expected):
    with pytest.raises(YAMLSyntaxError) as excinfo:
        parse_only(constraints, vel_min)
    assert expected in str(excinfo.value)


CONDITIONAL = """    manual_mode_speed_limit:
      when:
        eq: [$navigation_mode, "manual"]
      require:
        lt_eq: [$vel_max, 2.0]
      description: "in manual mode, vel_max must not exceed 2.0"
"""

SHORTHAND = """    manual_mode_speed_limit:
      when:
        $navigation_mode: "manual"
      require:
        lt_eq: [$vel_max, 2.0]
"""


def test_cpp_conditional_nests_the_requirement_inside_the_guard():
    code = generate('cpp', CONDITIONAL)
    guard = 'if (updated_params.navigation_mode == "manual") {'
    assert code.count(guard) == 2, 'expected a guard in update and declare'
    body = code[code.index(guard) :]
    assert body.index('if (!(updated_params.vel_max <= 2.0))') < body.index(
        'tl::make_unexpected'
    )
    # the guard is named in the error so the active threshold is explained
    assert "with navigation_mode == 'manual', vel_max ({}) must be <= 2.0" in code


def test_python_conditional_generates_valid_module():
    code = generate('python', CONDITIONAL)
    ast.parse(code)
    assert code.count('if updated_params.navigation_mode == "manual":') == 2
    assert code.count('if not (updated_params.vel_max <= 2.0):') == 2


def test_guard_shorthand_matches_the_explicit_form():
    # description does not reach the generated code, so the two must be identical
    assert generate('cpp', SHORTHAND) == generate('cpp', CONDITIONAL)


def test_inactive_guard_makes_the_requirement_vacuous_at_defaults():
    # navigation_mode defaults to autonomous, so the requirement cannot be
    # folded even though vel_max (1.6) would violate it under manual mode
    parse_only(
        '    c:\n      when:\n        $navigation_mode: "manual"\n'
        '      require:\n        lt_eq: [$vel_max, 1.0]\n'
    )


def test_active_guard_folds_the_requirement_at_defaults():
    with pytest.raises(YAMLSyntaxError) as excinfo:
        parse_only(
            '    c:\n      when:\n        $navigation_mode: "manual"\n'
            '      require:\n        lt_eq: [$vel_max, 1.0]\n',
            nav_mode='manual',
        )
    assert 'violated by the declared default values' in str(excinfo.value)
    assert "with navigation_mode == 'manual'" in str(excinfo.value)


@pytest.mark.parametrize(
    ('constraints', 'expected'),
    [
        (
            '    bad:\n      when:\n        $navigation_mode: "manual"\n',
            "has 'when' without 'require'",
        ),
        (
            '    bad:\n      require:\n        lt: [$vel_min, $vel_max]\n',
            "has 'require' without 'when'",
        ),
        (
            '    bad:\n      lt: [$vel_min, $vel_max]\n      when:\n'
            '        $navigation_mode: "manual"\n      require:\n'
            '        lt_eq: [$vel_max, 2.0]\n',
            'mixes',
        ),
        (
            '    bad:\n      when:\n        $navigation_mode: "teleop"\n'
            '      require:\n        lt_eq: [$vel_max, 2.0]\n',
            'can never be true',
        ),
        (
            '    bad:\n      when:\n        navigation_mode: "manual"\n'
            '      require:\n        lt_eq: [$vel_max, 2.0]\n',
            "unknown operator 'navigation_mode' in 'when'",
        ),
        (
            '    bad:\n      when:\n        eq: [$navigation_mode, "manual"]\n'
            '      require:\n        lt_eq: [$vel_max]\n',
            'takes exactly 2 operands',
        ),
    ],
)
def test_invalid_conditional_constraints_are_rejected(constraints, expected):
    with pytest.raises(YAMLSyntaxError) as excinfo:
        parse_only(constraints)
    assert expected in str(excinfo.value)


CUSTOM = """    srdf_matches_urdf:
      "my_project::srdf_matches_urdf": [$navigation_mode, $joints, 0.25]
      description: "an arbitrary predicate over several parameters"
"""

HEADER = 'my_project/validators.hpp'


def test_cpp_custom_constraint_calls_the_user_function():
    code = generate('cpp', CUSTOM, header=HEADER)
    call = (
        'my_project::srdf_matches_urdf(updated_params.navigation_mode, '
        'updated_params.joints, 0.25)'
    )
    assert code.count(call) == 2, 'expected a call in update and declare'
    # the user function supplies the message body
    assert (
        'fmt::format("Constraint \'srdf_matches_urdf\' violated: {}", '
        'validation_result.error())' in code
    )


def test_custom_constraint_accepts_array_and_string_operands():
    # arrays are rejected for built-in relations but are the point of a
    # custom predicate, so only references escape the scalar restriction
    code = generate('cpp', CUSTOM, header=HEADER)
    assert 'updated_params.joints' in code


def test_python_custom_constraint_uses_the_import_alias():
    constraints = '    c:\n      "custom_validators::check": [$vel_min, $vel_max]\n'
    code = generate('python', constraints, header='my_project.validators')
    ast.parse(code)
    assert code.count('custom_validators.check(') == 2
    assert "\"Constraint 'c' violated: \" + validation_result" in code


def test_custom_constraint_may_be_guarded():
    constraints = (
        '    c:\n      when:\n        $navigation_mode: "manual"\n'
        '      require:\n        "my_project::check": [$vel_max]\n'
    )
    code = generate('cpp', constraints, header=HEADER)
    guard = 'if (updated_params.navigation_mode == "manual") {'
    body = code[code.index(guard) :]
    assert body.index('my_project::check(updated_params.vel_max)') < body.index(
        'tl::make_unexpected'
    )
    assert "with navigation_mode == 'manual'" in code


def test_custom_constraint_is_not_folded_against_defaults():
    # vel_max would violate a built-in lt_eq here, but a predicate is opaque
    parse_only('    c:\n      "my_project::check": [$vel_max, 1.0]\n', header=HEADER)


@pytest.mark.parametrize(
    ('constraints', 'header', 'expected'),
    [
        (
            '    bad:\n      "my_project::check": [$vel_min]\n',
            '',
            'no validation header was given',
        ),
        (
            '    bad:\n      "my_project::check": []\n',
            HEADER,
            'with no operands',
        ),
        (
            '    bad:\n      "my_project::check": [1.0, 2.0]\n',
            HEADER,
            'without referencing any parameter',
        ),
        (
            '    bad:\n      when:\n        "my_project::check": [$vel_min]\n'
            '      require:\n        lt: [$vel_min, $vel_max]\n',
            HEADER,
            'can only be used as the requirement',
        ),
        (
            '    bad:\n      not_qualified: [$vel_min, $vel_max]\n',
            HEADER,
            'must be fully qualified',
        ),
    ],
)
def test_invalid_custom_constraints_are_rejected(constraints, header, expected):
    with pytest.raises(YAMLSyntaxError) as excinfo:
        parse_only(constraints, header=header)
    assert expected in str(excinfo.value)


def test_python_rejects_a_namespace_other_than_the_import_alias():
    yaml_file = write_yaml('    c:\n      "my_project::check": [$vel_min]\n')
    try:
        with pytest.raises(YAMLSyntaxError) as excinfo:
            GenerateCode('python').parse(yaml_file, 'my_project.validators')
        assert "must be qualified with 'custom_validators::'" in str(excinfo.value)
    finally:
        os.unlink(yaml_file)
