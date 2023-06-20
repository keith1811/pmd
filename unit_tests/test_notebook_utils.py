from typing import Dict, Any

import os
import mock
import yaml
import pprint
import pytest
import pathlib

from modules.utils.notebook_utils import *


@pytest.fixture
def expected_env_vars():
    return {'VAR1': 'value1', 'VAR2': 'value2'}

@pytest.fixture
def expected_config():
    return {'c1': 'value1', 'c2': 'value2'}

def test_load_and_set_env_vars(expected_env_vars, monkeypatch):
    env = 'test'
    expected_env_vars_path = os.path.join(os.pardir, 'conf', env, f'.{env}.env')
    expected_base_data_vars_vars_path = os.path.join(os.pardir, 'conf', '.base_data_params.env')
    
    # Mock the dotenv.load_dotenv method
    with mock.patch('dotenv.load_dotenv') as mock_load_dotenv:
        # Set the return value of dotenv.load_dotenv to None
        mock_load_dotenv.return_value = None

        # Set the mock environment variables using monkeypatch
        for k, v in expected_env_vars.items():
            monkeypatch.setenv(k, v)

        # Call the function being tested
        result = load_and_set_env_vars(env)

        # Assert that dotenv.load_dotenv was called with the expected arguments
        mock_load_dotenv.assert_has_calls([mock.call(expected_env_vars_path), 
                                           mock.call(expected_base_data_vars_vars_path)])

    # Get the environment variables loaded from the .env files
    env_vars = {k: v for k, v in result.items() if k in expected_env_vars}

    # Assert that the function loaded the expected environment variables
    assert env_vars == expected_env_vars
       
def test_load_config(expected_config):
    pipeline_name = 'test_pipeline'
    expected_config_path = os.path.join(os.pardir, 'conf', 'pipeline_configs', f'{pipeline_name}.yml')

    # Mock the pathlib.Path.read_text method to return the expected YAML config string
    with mock.patch.object(pathlib.Path, 'read_text') as mock_read_text:
        mock_read_text.return_value = yaml.dump(expected_config)

        # Call the function being tested
        result = load_config(pipeline_name, "test")

        # Assert that pathlib.Path.read_text was called with the expected argument
        mock_read_text.assert_called_once_with()

    # Assert that the function returned the expected config dictionary
    assert result == expected_config