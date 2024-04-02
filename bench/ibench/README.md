# AutoDBA - Vacuum Reinforcement Learning Project

## Overview
This project aims to enhance the PostgreSQL AutoVacuum feature using reinforcement learning (RL) techniques. By integrating a custom RL model with PostgreSQL, we seek to optimize vacuuming decisions dynamically based on database activity, improving performance and resource utilization.

## Components
- `autovac_driver.py`: Main script for the reinforcement learning driver that interacts with the PostgreSQL AutoVacuum feature.
- `iibench_driver.py`: iibench workload to simulate database load and test the RL model's effectiveness.
- `learning/`: Directory containing the reinforcement learning model and training scripts.

## Requirements
- Python 3.8+
- PyTorch
- PostgreSQL

## Installation
Provide step-by-step instructions to set up the project:

1. Clone the repository:
   ```
   git clone https://github.com/crystalcld/mytools
   ```
2. Navigate to the project directory:
   ```
   cd bench/ibench
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the `autovac_driver.py` script, you need to provide command line arguments for the mode and parameters. Here are the command structures for each mode:

### Benchmark Mode
```bash
python autovac_driver.py benchmark [max_episodes] [resume_id] [experiment_duration] [model_type] [model1_filename] [model2_filename] [instance_url] [instance_user] [instance_password] [instance_dbname]
```

### Learn Mode
```bash
python autovac_driver.py learn [max_episodes] [resume_id] [experiment_duration] [model_type] [model1_filename] [model2_filename] [instance_url] [instance_user] [instance_password] [instance_dbname]
```

Replace the bracketed terms with your actual values.

## Configuration
TODO

## License

This project is licensed under the MIT License.
