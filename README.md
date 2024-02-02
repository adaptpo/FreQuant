# FreQuant: A Reinforcement-Learning based Adaptive Portfolio Optimization with Multi-frequency Decomposition
This is a code for "FreQuant: A Reinforcement-Learning based Adaptive Portfolio Optimization with Multi-frequency Decomposition", submitted in KDD2024.

## Code Information
All codes are written by Python 3.8 and PyTorch 1.13.1.
This folder contains the code for FreQuant, an effective method for portfolio optimization using multi-frequency features. By utilizing historical price feature tensors of both assets and the market, FreQuant adeptly manages portfolios by explicitly incorporating multi-frequency features to identify individual assets. 



* The code of FreQuant is in `src` directory.
    * `main.py`: the main code which takes keyword arguments and initializes the market environment, trains the model accordingly.
    * `environment.py`: the code that defines the Market environment class. (_**Transaction Fee Implementation**_)
    * `network.py`: the code that defines the networks used in proposed method FreQuant. (_**FreQuant Architecture Implementation**_)
    * `normalizer.py`: the code that contains normalization methods for price signals. 
    * `ddpg.py`: the code that adopts DDPG algorithm explicitly for FreQuant. (_**Optimization Implementation**_)
    * `utils.py`: the code containing some util methods.
    * `mysql.py`: the code for reading/preprocessing asset price data, including the File I/O for the csv files.
    * `experiment.py`: the code for conducting experiments. Experiment class contains multiple useful methods for the analysis.
* The libraries/packages utilized by FreQuant are listed in the `requirements.txt` file. To create the corresponding Conda environment, you can use the command `conda create --name <your_env_name> --file requirements.txt`.

## Dataset Information
* You may download the dataset from Google Drive (~93.4Mb): https://drive.google.com/file/d/1rQptCS65znAmWFyGK1MO4HjUcoxAnLA_/view?usp=drive_link
  * Please unzip the files and place it under the data directory.
* The experiment utilized six real-world market datasets (U.S., KR, Crypto, CN, JP, U.K.).
* Three datasets are publicly accessible through the provided URLs. We have updated the data period and the number of assets existing in the dataset for the purpose of enabling extensive experimentation with the longer test period to check the robustness, and with a larger action space to check the ability for dynamic selection.  
  * The CN stock market dataset comprised **34** stocks from https://github.com/TradeMaster-NTU/TradeMaster
  * The JP stock market dataset comprised **118** stocks from https://datalab.snu.ac.kr/dtml/#datasets
  * The U.K. stock market dataset consists of **21** stocks from https://datalab.snu.ac.kr/dtml/#datasets
* Instead of relying on publicly accessible datasets, we employed three more datasets to assess the model's capacity to select valuable assets from a significantly large pool of market assets. Adhering to the data redistribution policies of the data sources, we are unable to publicly release these market datasets. 
  * The U.S. stock market dataset comprised **224** stocks and was sourced from https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/center-for-research-in-security-prices-crsp/
  * The KR stock market dataset encompassed **528** stocks and was gathered from https://finance.yahoo.com/
  * The Crypto market dataset encompassed **44** cryptocurrencies and was collected from https://coinmarketcap.com/
* The market dataset of FreQuant is in `data` directory.
  * `stocks_us.csv`: the U.S. stock market dataset.
    * `index_nyse.csv`: the NYSE index of the U.S. market (Primarily used).
    * `index_nasdaq.csv`: the Nasdaq index of the U.S. market (Secondary).
  * `crypto.csv`: the Crypto market dataset.
  * `stocks_ksp.csv`: the KR stock market dataset.
    * `index_ksp.csv`: the KOSPI index of the KR market.
  * `stocks_cn.csv`: the CN stock market dataset.
    * `index_cn.csv`: the SSE50 index of the CN market.
  * `stocks_jp.csv`: the JP stock market dataset. 
    * `index_jp.csv`: the NIKKEI225 index of the JP stock market.
  * `stocks_uk.csv`: the U.K. stock market dataset.
    * `index_uk.csv`: the FTSE100 index of the U.K. stock market.
  

  
## Execution Guideline
You can run and test the code in command-line interface (CLI) like terminal with the following examples:
   
   `python -u src/main.py` 
    will run everything in a default settings.

Or, 

   `python -u src/main.py --process train --max_epi 10 --use_FRE False --use_CTE True` will run with some of the specified keyword arguments.

Note that all the keyword arguments can be modified and list of available arguments are shown in `main.py`'s `add_arguments` methods. 

