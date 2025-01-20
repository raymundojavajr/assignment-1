from dagster import job, op

@op
def hello_op():
    return "Hello, Dagster!"

@job
def my_pipeline():
    hello_op()
