from modules.utils.config_utils import *

def test_get_model_name():
    model_name = get_model_name(env="ml")
    assert "po_classification" == model_name

def test_get_feature_concat_cols():
    exp_col_name_list_1 = ["SUPPLIER_ORG", "BUYER_ORG", "DESCRIPTION", "MANUFACTURER_NAME", "ITEM_TYPE", "IS_ADHOC"].sort()
    actual_col_name_list_1 = get_feature_concat_cols(env="ml",pipeline_name='model_train').sort()

    exp_col_name_list_2 = ["BUYER_ORG","SUPPLIER_ORG", "DESCRIPTION", "MANUFACTURER_NAME", "ITEM_TYPE", "IS_ADHOC"].sort()
    actual_col_name_list_2 = get_feature_concat_cols(env="notebookdev",pipeline_name='model_inference').sort()
    assert exp_col_name_list_1 == actual_col_name_list_1
    assert exp_col_name_list_2 == actual_col_name_list_2

def test_get_feature_concat_name():
    assert "CONCAT_FEATURE" == get_feature_concat_name(env="ml",pipeline_name='model_train')
    assert "CONCAT_FEATURE" == get_feature_concat_name(env="notebookdev",pipeline_name='model_inference')

def test_get_replacement_value():
    assert "N/A" == get_replacement_value(env="notebookdev")

def test_get_na_value_list():
    exp_na_values_list_1 = ["N/A","?","null", "NULL", "Null", "NA", "Not Available", "not available", "NAN", "nan","Nan"].sort()
    actual_na_values_list_1 = get_na_value_list(env="notebookdev").sort()
    assert actual_na_values_list_1 == exp_na_values_list_1