# Define the Databricks job
resource "databricks_job" "platform_machine_learning_workflow" {
  name = "platform_machine_learning_workflow_${var.env}"
  dynamic "job_cluster" {
      for_each = var.env == "notebookdev" ? []:["1"]
      content{
        job_cluster_key = "platform_machine_learning_workflow_cluster_${var.env}"
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
            first_on_demand       = 1
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
    schedule {
        quartz_cron_expression = var.platform_datalake_workflow_parameters.quartz_cron_expression
        timezone_id = "UTC"
        pause_status = "UNPAUSED"
    }
    git_source {
      url = var.git_source_url
      provider = var.git_source_provider
      commit = var.git_source_commit
    }
    tags = {
      repo = "ml-demandtrends"
      job_name = "platform_machine_learning_workflow"
      env = var.env
      run_type = "batch"
      trigger_type = "automatic"
    }
    task {
      task_key = "ml_supplier_po_item_record_model_task"
      notebook_task {
        notebook_path = "notebooks/run_batch_job_notebook"
        source = "GIT"
        base_parameters = {
          tasklist = "ml_supplier_po_item_record_model_module"
          env = var.env
          import_module = "ml_supplier_po_item_record_model_module"
        }
      }
      job_cluster_key = var.env == "notebookdev" ? null : "platform_machine_learning_workflow_cluster_${var.env}"
      existing_cluster_id = var.env == "notebookdev" ? one(data.databricks_clusters.notebookdev[0].ids) : null
      max_retries = 2
      min_retry_interval_millis = 300000
      retry_on_timeout = true
      timeout_seconds = var.platform_datalake_workflow_parameters.timeout_seconds
    }
    task {
      task_key = "ml_supplier_po_item_extract_task"
      depends_on {
        task_key = "ml_supplier_po_item_record_model_task"
      }
      notebook_task {
        notebook_path = "notebooks/run_batch_job_notebook"
        source = "GIT"
        base_parameters = {
          tasklist = "ml_supplier_po_item_extract_module"
          env = var.env
          import_module = "ml_supplier_po_item_extract_module"
        }
      }
      job_cluster_key = var.env == "notebookdev" ? null : "platform_machine_learning_workflow_cluster_${var.env}"
      existing_cluster_id = var.env == "notebookdev" ? one(data.databricks_clusters.notebookdev[0].ids) : null
      max_retries = 2
      min_retry_interval_millis = 300000
      retry_on_timeout = true
      timeout_seconds = var.platform_datalake_workflow_parameters.timeout_seconds
    }
    task {
      task_key = "ml_supplier_po_item_eval_descr_task"
      depends_on {
        task_key = "ml_supplier_po_item_extract_task"
      }
      notebook_task {
        notebook_path = "notebooks/run_batch_job_notebook"
        source = "GIT"
        base_parameters = {
            tasklist = "ml_supplier_po_item_eval_descr_module"
            env = var.env
            import_module = "ml_supplier_po_item_eval_descr_module"
        }
      }
      job_cluster_key = var.env == "notebookdev" ? null : "platform_machine_learning_workflow_cluster_${var.env}"
      existing_cluster_id = var.env == "notebookdev" ? one(data.databricks_clusters.notebookdev[0].ids) : null
      max_retries = 2
      min_retry_interval_millis = 300000
      retry_on_timeout = true
      timeout_seconds = var.platform_datalake_workflow_parameters.timeout_seconds
    }
    task {
      task_key = "ml_supplier_po_item_gds_export_task"
      depends_on {
        task_key = "ml_supplier_po_item_eval_descr_task"
      }
      notebook_task {
        notebook_path = "notebooks/run_batch_job_notebook"
        source = "GIT"
        base_parameters = {
          tasklist = ""
          env = var.env
          import_module = "ml_supplier_po_item_gds_export_module"
        }
      }
      job_cluster_key = var.env == "notebookdev" ? null : "platform_machine_learning_workflow_cluster_${var.env}"
      existing_cluster_id = var.env == "notebookdev" ? one(data.databricks_clusters.notebookdev[0].ids) : null
      max_retries = 2
      min_retry_interval_millis = 300000
      retry_on_timeout = true
      timeout_seconds = var.platform_datalake_workflow_parameters.timeout_seconds
    }
    task {
      task_key = "ml_supplier_po_item_gds_import_task"
      depends_on {
        task_key = "ml_supplier_po_item_gds_export_task"
      }
      notebook_task {
        notebook_path = "notebooks/run_batch_job_notebook"
        source = "GIT"
        base_parameters = {
          tasklist = ""
          env = var.env
          import_module = "ml_supplier_po_item_gds_import_module"
        }
      }
      job_cluster_key = var.env == "notebookdev" ? null : "platform_machine_learning_workflow_cluster_${var.env}"
      existing_cluster_id = var.env == "notebookdev" ? one(data.databricks_clusters.notebookdev[0].ids) : null
      max_retries = 2
      min_retry_interval_millis = 300000
      retry_on_timeout = true
      timeout_seconds = var.platform_datalake_workflow_parameters.timeout_seconds
    }
    task {
      task_key = "ml_supplier_po_item_inference_task"
      depends_on {
        task_key = "ml_supplier_po_item_gds_import_task"
      }
      notebook_task {
        notebook_path = "notebooks/run_inference_job_notebook"
        source = "GIT"
        base_parameters = {
          tasklist = "ml_supplier_po_item_inference_module"
          env = var.env
          import_module = "ml_supplier_po_item_inference_module"
        }
      }
      job_cluster_key = var.env == "notebookdev" ? null : "platform_machine_learning_workflow_cluster_${var.env}"
      existing_cluster_id = var.env == "notebookdev" ? one(data.databricks_clusters.notebookdev[0].ids) : null
      max_retries = 2
      min_retry_interval_millis = 300000
      retry_on_timeout = true
      timeout_seconds = var.platform_datalake_workflow_parameters.timeout_seconds
    }
    task {
      task_key = "ml_supplier_po_item_gds_update_task"
      depends_on {
        task_key = "ml_supplier_po_item_inference_task"
      }
      notebook_task {
        notebook_path = "notebooks/run_batch_job_notebook"
        source = "GIT"
        base_parameters = {
          tasklist = "ml_supplier_po_item_gds_update_module"
          env = var.env
          import_module = "ml_supplier_po_item_gds_update_module"
        }
      }
      job_cluster_key = var.env == "notebookdev" ? null : "platform_machine_learning_workflow_cluster_${var.env}"
      existing_cluster_id = var.env == "notebookdev" ? one(data.databricks_clusters.notebookdev[0].ids) : null
      max_retries = 2
      min_retry_interval_millis = 300000
      retry_on_timeout = true
      timeout_seconds = var.platform_datalake_workflow_parameters.timeout_seconds
    }
    task {
      task_key = "ml_supplier_po_item_finalize_task"
      depends_on {
        task_key = "ml_supplier_po_item_gds_update_task"
      }
      notebook_task {
        notebook_path = "notebooks/run_batch_job_notebook"
        source = "GIT"
        base_parameters = {
          tasklist = "ml_supplier_po_item_finalize_module"
          env = var.env
          import_module = "ml_supplier_po_item_finalize_module"
        }
      }
      job_cluster_key = var.env == "notebookdev" ? null : "platform_machine_learning_workflow_cluster_${var.env}"
      existing_cluster_id = var.env == "notebookdev" ? one(data.databricks_clusters.notebookdev[0].ids) : null
      max_retries = 2
      min_retry_interval_millis = 300000
      retry_on_timeout = true
      timeout_seconds = var.platform_datalake_workflow_parameters.timeout_seconds
    }
    task {
      task_key = "ml_supplier_po_item_load_task"
      depends_on {
        task_key = "ml_supplier_po_item_finalize_task"
      }
      notebook_task {
        notebook_path = "notebooks/run_batch_job_notebook"
        source = "GIT"
        base_parameters = {
          tasklist = "ml_supplier_po_item_load_module"
          env = var.env
          import_module = "ml_supplier_po_item_load_module"
        }
      }
      job_cluster_key = var.env == "notebookdev" ? null : "platform_machine_learning_workflow_cluster_${var.env}"
      existing_cluster_id = var.env == "notebookdev" ? one(data.databricks_clusters.notebookdev[0].ids) : null
      max_retries = 2
      min_retry_interval_millis = 300000
      retry_on_timeout = true
      timeout_seconds = var.platform_datalake_workflow_parameters.timeout_seconds
    }
    task {
      task_key = "ml_supplier_po_item_aggregate_task"
      depends_on {
        task_key = "ml_supplier_po_item_load_task"
      }
      notebook_task {
        notebook_path = "notebooks/run_batch_job_notebook"
        source = "GIT"
        base_parameters = {
          tasklist = "ml_supplier_po_item_aggregate_module"
          env = var.env
          import_module = "ml_supplier_po_item_aggregate_module"
        }
      }
      job_cluster_key = var.env == "notebookdev" ? null : "platform_machine_learning_workflow_cluster_${var.env}"
      existing_cluster_id = var.env == "notebookdev" ? one(data.databricks_clusters.notebookdev[0].ids) : null
      max_retries = 2
      min_retry_interval_millis = 300000
      retry_on_timeout = true
      timeout_seconds = var.platform_datalake_workflow_parameters.timeout_seconds
    }
}