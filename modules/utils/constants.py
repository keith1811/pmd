from enum import Enum

class MLTable(Enum):
    ENRICHMENT_ML_SUPPLIER_PO_ITEM = 'ml_supplier_po_item'
    ENRICHMENT_ML_GDS_UNSPSC_REPORT = "ml_gds_unspsc_report"
    GENERAL_ML_MODEL_INFO = 'ml_model_info'
    STAGING_ML_UNSPSC_REPORT_GDS = "ml_unspsc_report_gds"
    CONSUMPTION_FACT_SUPPLIER_PO_ITEM = "fact_supplier_po_item"
    CONSUMPTION_NETWORK_SUPPLIER_PO_ITEM = "fact_network_po_item"
    CONSUMPTION_DIM_ML_SUPPLIER_PO_ITEM = 'dim_ml_supplier_po_item'
    RAW_COMMODITY = "commodity"
    CONSUMPTION_COMMODITY = "dim_directory_commodity"
    TRAINING_ML_BUYER_PO_ITEM = "ml_buyer_po_item"
    RAW_ML_BUYER_PO_ITEM = "ml_buyer_po_item"
    ENRICHMENT_ML_BNA_UPSPSC_REPORT = 'ml_bna_unspsc_report'