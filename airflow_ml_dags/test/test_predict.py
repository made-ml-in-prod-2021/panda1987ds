def test_dag_loaded(dag_bag):
    dag = dag_bag.dags.get('predict')
    assert dag_bag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 2
