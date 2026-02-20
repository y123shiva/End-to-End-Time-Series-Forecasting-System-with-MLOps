#!/bin/bash
# Test script to verify Airflow DAG can execute

echo "=========================================="
echo "Airflow DAG Test: financial_forecast_training"
echo "=========================================="
echo ""

# Set Airflow home
export AIRFLOW_HOME=/workspaces/Time-Series-Forecasting/airflow
export PYTHONPATH=/workspaces/Time-Series-Forecasting:$PYTHONPATH

echo "✓ Environment variables set"
echo "  AIRFLOW_HOME: $AIRFLOW_HOME"
echo "  PYTHONPATH: $PYTHONPATH"
echo ""

# Verify Python can import the DAG
echo "Testing DAG import..."
python3 -c "
import sys
sys.path.insert(0, '/workspaces/Time-Series-Forecasting')
from dags.forecast_dag import dag
print('✅ DAG imports successfully')
print(f'   DAG ID: {dag.dag_id}')
print(f'   Tasks: {[t.task_id for t in dag.tasks]}')
print(f'   Schedule: {dag.schedule_interval}')
" 2>&1 || exit 1

echo ""
echo "Testing manual DAG execution (training script)..."
echo "Running: python -m src.pipelines.train"
echo ""

# Change to workspace and run training
cd /workspaces/Time-Series-Forecasting
python -m src.pipelines.train 2>&1 | grep -E "(✅|❌|🏆|RMSE|MAE|Best Model)"

echo ""
echo "=========================================="
echo "✅ DAG execution test complete!"
echo "=========================================="
