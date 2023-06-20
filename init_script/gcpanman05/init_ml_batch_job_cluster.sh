#!/bin/bash

pip install platform-datalake-jobs -i https://$PIP_USR:$PIP_PWD@common.repositories.cloud.sap/artifactory/api/pypi/build.releases.pypi/simple

sed -i '/^PIP_USR/d' /databricks/spark/conf/spark-env.sh
sed -i '/^PIP_PWD/d' /databricks/spark/conf/spark-env.sh