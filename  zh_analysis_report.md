# Chinese Narrative Analysis Report

## Data Collection
Chinese-language news articles were extracted from the `intfloat/multilingual_cc_news` dataset via HuggingFace streaming. Using a set of Traditional Chinese keywords covering protest, corruption, and economic grievance themes, 1,318,149 articles were matched out of 6,133,244 total articles scanned (match rate ~21%). After removing articles with missing publication dates, 1,290,663 articles with valid timestamps remained, spanning 2010 to 2021.

## Narrative Classification
Articles were classified into three narrative categories using keyword matching: protest narratives (抗議, 示威, 暴動, etc.), corruption narratives (腐敗, 貪污, 賄賂, etc.), and economic narratives (緊縮, 能源, 通貨膨脹, etc.). Daily article counts per category were aggregated into a time series. The distribution was: protest (65,592 articles), economic (87,695 articles), and corruption (19,033 articles).

## Time Series Overview
The 2018–2021 window contained the highest data density and was selected for VAR modelling. A 7-day rolling average was applied to smooth daily fluctuations. A clear protest spike is visible in mid-2019 (peaking at ~175 articles/day), consistent with the Hong Kong Anti-Extradition Law Amendment Bill protests. Economic narratives showed a steady upward trend throughout 2020–2021, likely reflecting trade war pressures and rising inflation.

## VAR Model & Granger Causality
A Vector Autoregression (VAR) model with 5 lags was fitted on first-differenced data (1,449 observations). Granger causality tests were conducted in three directions:

| Direction | p-value | Result |
|---|---|---|
| Economic → Protest | 0.0629 | Not significant |
| Corruption → Protest | 0.5516 | Not significant |
| Protest → Economic | 0.5831 | Not significant |

None of the tested directions reached statistical significance at the 0.05 level, though the Economic → Protest direction approached the threshold (p=0.063).

## Impulse Response Analysis
Impulse response functions (IRF) over 14 days showed that a shock to economic narratives produces a small negative response in protest narratives, bottoming out around day 2 before returning to zero by day 10. A similar pattern was observed for corruption shocks. In both cases, the confidence intervals crossed zero throughout, confirming no statistically reliable lead-lag relationship.

## Discussion
The residual correlation matrix revealed moderate positive correlations between all three narrative pairs (0.30–0.39), suggesting that protest, corruption, and economic narratives tend to co-occur simultaneously in Chinese-language news rather than following a sequential pattern. This may reflect the nature of Chinese-language media coverage, where these themes are frequently reported together within the same event context rather than as separate developments. This finding contrasts with the expected grievance escalation sequence (economic hardship → corruption framing → protest mobilisation) observed in other language contexts.

## Summary
This study systematically analysed 6.13 million Chinese-language news articles, examining the dynamic relationships between three narrative categories — economic grievance, corruption, and protest mobilisation — through keyword classification, time series construction, and VAR modelling. The results show that within the 2018–2021 analysis window, no statistically significant Granger causal relationships were found among the three narratives, meaning that changes in any one narrative category could not reliably predict subsequent movements in another. However, the moderate contemporaneous correlations between all three pairs (0.30–0.39) suggest that these narratives tend to co-occur simultaneously in Chinese-language media, rather than following the linear logic of "economic pressure triggers corruption framing, which in turn drives protest mobilisation." The close alignment between the 2019 protest spike and the Hong Kong political events further indicates that politically-driven triggers may carry greater explanatory power in this context than economic or corruption narratives.

