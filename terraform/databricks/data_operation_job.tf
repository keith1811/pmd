# Define the Databricks job
resource "databricks_job" "data_operation_job" {
    count = var.env == "prod" ? 0 : 1
    name = "ml_data_operation_job_${var.env}"
    dynamic "job_cluster" {
      for_each = var.env == "notebookdev" ? []:["1"]
      content{
        job_cluster_key = "ml_data_operation_job_cluster_${var.env}"
        new_cluster {
          instance_pool_id        = var.env == "notebookdev" ? null : data.databricks_instance_pool.pool[0].id
          num_workers             = 1
          spark_version           = data.databricks_spark_version.lts_runtime.id
          data_security_mode      = "LEGACY_SINGLE_USER_STANDARD"
          autotermination_minutes = 0
          runtime_engine          = "STANDARD"
          spark_conf = {
            "spark.databricks.delta.preview.enabled" : true
          }
          spark_env_vars = {
            "PIP_USR": "{{secrets/bna-adbscope-common-repo/pip_usr}}"
            "PIP_PWD": "{{secrets/bna-adbscope-common-repo/pip_pwd}}"
          }
          init_scripts {
            dbfs {
              destination = "dbfs:/init_script/${var.env}/init_ml_batch_job_cluster.sh"
            }
          }
          azure_attributes {
            first_on_demand        = 1
            availability          = "ON_DEMAND_AZURE"
            spot_bid_max_price    = -1
          }
        }
      }
    }
    email_notifications {
        no_alert_for_skipped_runs = false
    }
    timeout_seconds = 0
    max_concurrent_runs = 1
    always_running = false
    task {
            task_key = "DataOperationJob"
            notebook_task {
                notebook_path = "notebooks/run_data_operation_notebook"
                source = "GIT"
                base_parameters = {
                    env = var.env
                }
            }
            job_cluster_key = var.env == "notebookdev" ? null : "ml_data_operation_job_cluster_${var.env}"
            existing_cluster_id = var.env == "notebookdev" ? one(data.databricks_clusters.notebookdev[0].ids) : null
            max_retries = 2
            min_retry_interval_millis = 300000
            retry_on_timeout = true
            timeout_seconds = 600
    }
    git_source {
        url = var.git_source_url
        provider = var.git_source_provider
        commit = var.git_source_commit
    }
}