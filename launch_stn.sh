python -m domainbed.scripts.sweep launch \
       --data_dir=${HOME}/data/domainbed \
       --output_dir=output_train \
       --command_launcher multi_gpu \
       --algorithms STN \
       --datasets PACS VLCS OfficeHome TerraIncognita DomainNet \
       --single_test_envs \
       --n_hparams 1 \
       --n_trials 3
