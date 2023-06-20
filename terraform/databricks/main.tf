data "databricks_spark_version" "lts_runtime" {
  long_term_support = true
  spark_version = "3.3.2"
  scala = "2.12"
}

data "databricks_clusters" "notebookdev" {
  count = var.env == "notebookdev" ? 1 : 0
  cluster_name_contains = "notebookdev"
}

data "databricks_clusters" "streaming" {
  count = var.env == "notebookdev" ? 0 : 1
  cluster_name_contains = "streaming"
}

data "databricks_instance_pool" "pool" {
  count = var.env == "notebookdev" ? 0 : 1
  name = "bna-${var.landscape}-adbpool-shared"
}

# data "databricks_instance_pool" "streaming" {
#   count = var.env == "notebookdev" ? 0 : 1
#   name = "bna-${var.landscape}-adbpool-streaming-shared"
# }