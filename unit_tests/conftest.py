"""
This conftest.py contains handy components that prepare SparkSession and other relevant objects.
"""

import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator
from unittest.mock import patch

import mlflow
import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
import logging
from dataclasses import dataclass


@dataclass
class FileInfoFixture:
    """
    This class mocks the DBUtils FileInfo object
    """

    path: str
    name: str
    size: int
    modificationTime: int


class DBUtilsFixture:
    """
    This class is used for mocking the behaviour of DBUtils inside tests.
    """

    def __init__(self):
        self.fs = self
        self.widgets = self

    def cp(self, src: str, dest: str, recurse: bool = False):
        copy_func = shutil.copytree if recurse else shutil.copy
        copy_func(src, dest)

    def ls(self, path: str):
        _paths = Path(path).glob("*")
        _objects = [
            FileInfoFixture(str(p.absolute()), p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in _paths
        ]
        return _objects

    def mkdirs(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)

    def mv(self, src: str, dest: str, recurse: bool = False):
        copy_func = shutil.copytree if recurse else shutil.copy
        shutil.move(src, dest, copy_function=copy_func)

    def put(self, path: str, content: str, overwrite: bool = False):
        _f = Path(path)

        if _f.exists() and not overwrite:
            raise FileExistsError("File already exists")

        _f.write_text(content, encoding="utf-8")

    def rm(self, path: str, recurse: bool = False):
        if os.path.exists(path):
            deletion_func = shutil.rmtree if recurse else os.remove
            deletion_func(path)

    def get(self, name: str):
        if name == "need_archive":
            return "true"
        return None

@pytest.fixture(scope="session")
def base_artifact_path():
    return "tmp/training"

@pytest.fixture(scope="module")
def base_testdata_path(request):
    return f"tmp/testdata/{request.node.name}"

@pytest.fixture(scope="session")
def spark(base_artifact_path) -> SparkSession:
    logging.info("Configuring Spark session for testing environment")
    warehouse_dir = os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/{base_artifact_path}/warehouse")
    logging.info("Warehouse Dir is " + warehouse_dir)
    _builder = (
        SparkSession.builder#.master("local[1]")
        .config("spark.hive.metastore.warehouse.dir", Path(warehouse_dir).as_uri())
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    spark: SparkSession = configure_spark_with_delta_pip(_builder).getOrCreate()
    logging.info("Spark session configured")
    yield spark
    logging.info("Shutting down Spark session")
    spark.stop()
    if Path(warehouse_dir).exists():
        shutil.rmtree(warehouse_dir)


@pytest.fixture(scope="session", autouse=True)
def mlflow_local(base_artifact_path):
    logging.info("Configuring local MLflow instance")
    tracking_uri = os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/{base_artifact_path}/tracking")
    registry_uri = os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/{base_artifact_path}/registry")
    logging.info("Tracking Uri is " + tracking_uri)
    logging.info("Registry Uri is " + registry_uri)
    mlflow.set_tracking_uri(Path(tracking_uri))
    mlflow.set_registry_uri(registry_uri)
    logging.info("MLflow instance configured")
    yield None

    mlflow.end_run()

    if Path(tracking_uri).exists():
        shutil.rmtree(tracking_uri)
    if Path(registry_uri).exists():
        shutil.rmtree(registry_uri)
    # if Path(registry_uri).exists():
    #     Path(registry_uri).unlink()
    logging.info("Test session finished, unrolling the MLflow instance")

@pytest.fixture(scope="session")
def dbutils():
    dbutils = DBUtilsFixture()
    dbutils.get_dbutils = lambda _: DBUtilsFixture()
    return dbutils

@pytest.fixture(scope="session", autouse=True)
def dbutils_mocker() -> Iterator[None]:
    with patch("utils.sbnutils.DBUtils") as dbutils_fixture:
        dbutils_fixture.return_value = DBUtilsFixture()
        dbutils_fixture().get_dbutils = lambda _: DBUtilsFixture()
        yield dbutils_fixture
@pytest.fixture(scope="session", autouse=True)
def get_dbutils_mocker() -> Iterator[None]:
    with patch("utils.sbnutils.get_dbutils") as dbutils:
        dbutils.return_value = DBUtilsFixture()
        yield dbutils

@pytest.fixture(scope="session", autouse=True)
def get_job_name_mocker():
    with patch("utils.sbnutils._get_job_name", return_value = "test") as job_name:
        yield job_name

@pytest.fixture(scope="session", autouse=True)
def dbutils_feature_mocker() -> Iterator[None]:
    with patch("modules.model_train_feature_creator_module.DBUtils") as dbutils_fixture:
        dbutils_fixture.return_value = DBUtilsFixture()
        dbutils_fixture().get_dbutils = lambda _: DBUtilsFixture()
        yield dbutils_fixture
@pytest.fixture(scope="session", autouse=True)
def spark_feature_mocker(spark) -> SparkSession:
    with patch("modules.model_train_feature_creator_module.SparkSession.builder.getOrCreate") as getOrCreate:
        getOrCreate.return_value = spark
        yield getOrCreate

@pytest.fixture(scope="session", autouse=True)
def dbutils_train_mocker() -> Iterator[None]:
    with patch("modules.model_train_module.DBUtils") as dbutils_fixture:
        dbutils_fixture.return_value = DBUtilsFixture()
        dbutils_fixture().get_dbutils = lambda _: DBUtilsFixture()
        yield dbutils_fixture
@pytest.fixture(scope="session", autouse=True)
def spark_train_mocker(spark) -> SparkSession:
    with patch("modules.model_train_module.SparkSession.builder.getOrCreate") as getOrCreate:
        getOrCreate.return_value = spark
        yield getOrCreate

@pytest.fixture(scope="session", autouse=True)
def dbutils_inference_mocker() -> Iterator[None]:
    with patch("modules.model_inference_module.DBUtils") as dbutils_fixture:
        dbutils_fixture.return_value = DBUtilsFixture()
        dbutils_fixture().get_dbutils = lambda _: DBUtilsFixture()
        yield dbutils_fixture
@pytest.fixture(scope="session", autouse=True)
def spark_inference_mocker(spark) -> SparkSession:
    with patch("modules.model_inference_module.SparkSession.builder.getOrCreate") as getOrCreate:
        getOrCreate.return_value = spark
        yield getOrCreate

@pytest.fixture(scope="module", autouse=True)
def change_test_dir(request, monkeymodule):
    monkeymodule.chdir(request.fspath.dirname)

@pytest.fixture(scope="module")
def current_test_dir(request):
    test_dir = os.path.dirname(request.module.__file__)
    logging.info("Current Test Dir is " + test_dir)
    return test_dir

@pytest.fixture(scope='module')
def monkeymodule():
    with pytest.MonkeyPatch.context() as mp:
        yield mp

# def pytest_collection_modifyitems(session, config, items):
#     for i in items:
#         print(i.path)
