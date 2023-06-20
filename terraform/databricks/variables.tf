variable "landscape" {
    type        = string
    default     = "dev"
}
variable "env" {
    type        = string
    default     = "notebookdev"
}
variable "git_source_commit" {
    type        = string
    default     = "HEAD"
}
variable "git_source_url" {
    type        = string
    default     = "https://github.tools.sap/BNA/platform-ml-demandtrends"
}
variable "git_source_provider" {
    type        = string
    default     = "gitHubEnterprise"
}
variable "platform_datalake_workflow_parameters" {
    type = object({
        quartz_cron_expression          = string
        timeout_seconds                 = number
    })
    default = {
        quartz_cron_expression          = "0 0 8 ? * * *"
        timeout_seconds                 = 21600
    }
}
