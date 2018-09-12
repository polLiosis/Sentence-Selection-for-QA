## Create QA pairs from BioASQ files

Script used in order to download and create the BioASQ dataset.

#### Specification
- **BioASQ_json_files**: Published json files.
- **Datasets**: Produced datasets.
- **PubMeds**: Data related to various PubMed documents. 
- **main.py**: First script to run. It downloads PubMeds and creates datasets.
- **dataset.py**: Script that creates datasets based on BioASQ format.
- **improve_train_set**: Run this script in case you want to improve a train set (based on bioASQ format).
- **improve_test_set**: Run this script in case you want to improve a dev/test set (based on BioASQ format).
- **download_nltk.py**: Run this script in order to download the nltk library.

