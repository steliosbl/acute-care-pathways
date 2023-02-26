from acd_experiment.systematic_comparison import construct_parser, run_experiment
from acd_experiment.sci import SCIData, SCICols

# from acd_experiment.salford_adapter import SalfordAdapter
from datasets.salford import SalfordData
import pandas as pd

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

if __name__ == "__main__":
    parser = construct_parser()
    logging.info("Constructing SCI dataset")
    scii = (
        SCIData(
            SCIData.quickload("data/SCI/sci_processed.h5").sort_values(
                "AdmissionDateTime"
            )
        )
        .mandate(SCICols.news_data_raw)
        .derive_ae_diagnosis_stems(onehot=False)
    )
    # logging.info("Constructing Salford dataset")
    # sal = pd.read_hdf(f'data/Salford/sal_processed.h5', 'table')
    logging.info("Starting experiment")
    run_experiment(parser.parse_args(), scii)
