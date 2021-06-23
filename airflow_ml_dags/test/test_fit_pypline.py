def test_dag_loaded(dag_bag):
    dag = dag_bag.dags.get('fit_pypline')
    print(f'qqqqqqq {dag.tasks}')
    assert dag_bag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 5
    assert dag.tasks[0].task_id == 'wait_for_file'
    assert dag.tasks[1].task_id == 'preprocess'
    assert dag.tasks[2].task_id == 'split'
    assert dag.tasks[3].task_id == 'fit'
    assert dag.tasks[4].task_id == 'validate'
