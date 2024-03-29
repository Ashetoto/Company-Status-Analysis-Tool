# Financial-Data-Analysis-Tool-test
The dataset used in this project comprises quarterly financial data of A-share listed companies from 2017 to 2022, collected from the CSMAR Futures and Stock Analysis Trading Database. 

As the banking industry is relatively unique, this experiment's dataset consists of A-share listed companies excluding the banking industry. 67 features were selected using the Pearson correlation coefficient to screen financial indicators. Then, some company data with too many missing values and insufficient financial report data (22 samples, including data from every March, June, September, and December from 2017 to 2021, and data from June and December 2022) were deleted. 

For individual missing values, the average value of the adjacent quarters was used for filling, and eventually, data from 3175 listed companies were retained. The financial report data of enterprises were analyzed from eight dimensions, including cash flow analysis, solvency, operating capacity, profitability, development capacity, per-share indicators, risk level, and violation situation. The entropy method was used to weigh the financial report data of enterprises and score them. Based on the distribution of scores, whether the quarterly financial situation of the company is abnormal was judged, and this was used as the data label for training and testing the dataset. 

This project innovatively combines the LSTM and cutting-edge graph neural network algorithm LUNAR to conduct risk detection of listed companies based on six years of financial indicators.
