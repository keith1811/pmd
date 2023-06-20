# Variables may not be used inside provider definition.
terraform {

  required_providers {

    databricks = {
      source  = "databricks/databricks"
      version = "1.13.0"
    }
  }
}
