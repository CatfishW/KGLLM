@echo off
REM Evaluate multiple checkpoints

echo ======================================================================
echo Evaluating Checkpoint 5
echo ======================================================================
python evaluate_with_metrics.py --checkpoint "outputs_multipath_relation_only_5/checkpoints/last.ckpt" --vocab "outputs_multipath_relation_only_5/vocab.json" --test_data "../Data/webqsp_final/test.parquet" --output "eval_results_checkpoint_5.json" --max_examples 200

echo ======================================================================
echo Evaluating Checkpoint 6
echo ======================================================================
python evaluate_with_metrics.py --checkpoint "outputs_multipath_relation_only_6/checkpoints/last.ckpt" --vocab "outputs_multipath_relation_only_6/vocab.json" --test_data "../Data/webqsp_final/test.parquet" --output "eval_results_checkpoint_6.json" --max_examples 200

echo ======================================================================
echo Evaluating Checkpoint 7
echo ======================================================================
python evaluate_with_metrics.py --checkpoint "outputs_multipath_relation_only_7/checkpoints/last.ckpt" --vocab "outputs_multipath_relation_only_7/vocab.json" --test_data "../Data/webqsp_final/test.parquet" --output "eval_results_checkpoint_7.json" --max_examples 200

echo ======================================================================
echo Running Entity Retrieval Evaluation on all checkpoints
echo ======================================================================
python evaluate_with_entity_retrieval.py --eval_results "eval_results_checkpoint_5.json" --test_data "../Data/webqsp_final/test.parquet" --vocab "outputs_multipath_relation_only_5/vocab.json" --output "entity_retrieval_checkpoint_5.json" --max_examples 200
python evaluate_with_entity_retrieval.py --eval_results "eval_results_checkpoint_6.json" --test_data "../Data/webqsp_final/test.parquet" --vocab "outputs_multipath_relation_only_6/vocab.json" --output "entity_retrieval_checkpoint_6.json" --max_examples 200
python evaluate_with_entity_retrieval.py --eval_results "eval_results_checkpoint_7.json" --test_data "../Data/webqsp_final/test.parquet" --vocab "outputs_multipath_relation_only_7/vocab.json" --output "entity_retrieval_checkpoint_7.json" --max_examples 200

echo ======================================================================
echo All evaluations complete!
echo ======================================================================
pause
