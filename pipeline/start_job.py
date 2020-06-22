from airflow.models import Variable
from time import gmtime, strftime


def start():
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    Variable.set("timestamp", timestamp_prefix)
