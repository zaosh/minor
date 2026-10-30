# Model comparison (mean ± std across walk-forward folds)

## Volatility (vol_fwd)

| model | rmse | mae | dir_acc | qlike | mse_var |
|---|---|---|---|---|---|
| combo_hg | 0.0717 ± 0.0493 | 0.0486 ± 0.0211 | 0.6672 ± 0.0815 | 0.3150 ± 0.3291 | 0.005382 ± 0.011984 |
| combo_hgx | 0.0697 ± 0.0491 | 0.0468 ± 0.0209 | 0.6712 ± 0.0813 | 0.2979 ± 0.3101 | 0.005287 ± 0.011809 |
| ewma | 0.0781 ± 0.0580 | 0.0554 ± 0.0372 | 0.6185 ± 0.0614 | 0.3454 ± 0.3515 | 0.006723 ± 0.014937 |
| garch | 0.0748 ± 0.0453 | 0.0524 ± 0.0184 | 0.6629 ± 0.0826 | 0.3092 ± 0.2753 | 0.005267 ± 0.011559 |
| gjr | 0.0734 ± 0.0416 | 0.0518 ± 0.0169 | 0.6674 ± 0.0868 | 0.2809 ± 0.2078 | 0.004969 ± 0.010793 |
| har | 0.0728 ± 0.0539 | 0.0485 ± 0.0256 | 0.6557 ± 0.0724 | 0.3612 ± 0.4147 | 0.005750 ± 0.012843 |
| ledoit_wolf | 0.0834 ± 0.0609 | 0.0621 ± 0.0410 | 0.6314 ± 0.0495 | 0.4616 ± 0.5435 | 0.006447 ± 0.014211 |
| lstm | 0.0718 ± 0.0458 | 0.0483 ± 0.0217 | 0.6817 ± 0.0632 | 0.2993 ± 0.2718 | 0.005183 ± 0.011277 |
| naive | 0.0834 ± 0.0590 | 0.0578 ± 0.0344 | nan ± nan | 0.3987 ± 0.3377 | 0.008211 ± 0.018211 |
| rf | 0.0724 ± 0.0532 | 0.0485 ± 0.0247 | 0.6583 ± 0.0605 | 0.3378 ± 0.3688 | 0.005800 ± 0.012969 |
| xgboost | 0.0710 ± 0.0496 | 0.0479 ± 0.0234 | 0.6694 ± 0.0580 | 0.3057 ± 0.2952 | 0.005474 ± 0.012173 |


## Drawdown >= 5% probability

| model | auc | brier | logloss |
|---|---|---|---|
| climatology | 0.5000 ± 0.0000 | 0.2199 ± 0.0351 | 0.6318 ± 0.0723 |
| combo_hg | 0.6260 ± 0.0653 | 0.2144 ± 0.0436 | 0.6175 ± 0.0948 |
| combo_hgx | 0.6342 ± 0.0538 | 0.2068 ± 0.0408 | 0.6012 ± 0.0881 |
| ewma | 0.6135 ± 0.0720 | 0.2237 ± 0.0676 | 0.6458 ± 0.1698 |
| fhs_mc | 0.6288 ± 0.0520 | 0.2050 ± 0.0461 | 0.5975 ± 0.1045 |
| fhs_mc_logit | 0.6288 ± 0.0520 | 0.2046 ± 0.0392 | 0.5973 ± 0.0844 |
| garch | 0.6316 ± 0.0550 | 0.2225 ± 0.0278 | 0.6371 ± 0.0605 |
| gjr | 0.6435 ± 0.0599 | 0.2196 ± 0.0280 | 0.6316 ± 0.0613 |
| har | 0.6101 ± 0.0739 | 0.2134 ± 0.0570 | 0.6135 ± 0.1282 |
| ledoit_wolf | 0.5882 ± 0.1331 | 0.2420 ± 0.0886 | 0.6849 ± 0.2127 |
| lstm | 0.6318 ± 0.0633 | 0.2062 ± 0.0464 | 0.6015 ± 0.1019 |
| naive | 0.6150 ± 0.0883 | 0.2229 ± 0.0646 | 0.6486 ± 0.1621 |
| rf | 0.6069 ± 0.0324 | 0.2061 ± 0.0478 | 0.5994 ± 0.1058 |
| xgboost | 0.6155 ± 0.0303 | 0.2101 ± 0.0452 | 0.6082 ± 0.0968 |


## 95% 1-day VaR (target exceedance 5%)

| model | pinball | exceedance | exc_range | kupiec_reject |
|---|---|---|---|---|
| combo_hg | 0.001392 ± 0.000521 | 0.0495 ± 0.0259 | 2.1%-8.6% | 5/6 |
| combo_hg_cal | 0.001397 ± 0.000537 | 0.0466 ± 0.0164 | 3.1%-7.6% | 5/6 |
| combo_hgx | 0.001384 ± 0.000545 | 0.0496 ± 0.0224 | 2.3%-8.2% | 5/6 |
| combo_hgx_cal | 0.001397 ± 0.000568 | 0.0466 ± 0.0152 | 3.1%-7.5% | 5/6 |
| ewma | 0.001400 ± 0.000554 | 0.0546 ± 0.0215 | 2.5%-8.1% | 5/6 |
| ewma_cal | 0.001421 ± 0.000576 | 0.0483 ± 0.0152 | 3.2%-7.4% | 5/6 |
| fhs | 0.001394 ± 0.000511 | 0.0529 ± 0.0269 | 2.4%-8.9% | 6/6 |
| fhs_cal | 0.001399 ± 0.000521 | 0.0473 ± 0.0169 | 3.0%-7.7% | 6/6 |
| garch | 0.001392 ± 0.000485 | 0.0463 ± 0.0253 | 2.0%-8.1% | 5/6 |
| garch_cal | 0.001395 ± 0.000512 | 0.0460 ± 0.0158 | 2.8%-7.3% | 5/6 |
| gjr | 0.001367 ± 0.000457 | 0.0471 ± 0.0252 | 2.0%-8.1% | 5/6 |
| gjr_cal | 0.001371 ± 0.000481 | 0.0460 ± 0.0152 | 2.9%-7.2% | 4/6 |
| har | 0.001415 ± 0.000590 | 0.0552 ± 0.0275 | 2.4%-9.6% | 6/6 |
| har_cal | 0.001414 ± 0.000579 | 0.0477 ± 0.0172 | 3.2%-7.9% | 4/6 |
| hist_var | 0.001479 ± 0.000681 | 0.0586 ± 0.0281 | 3.8%-11.1% | 6/6 |
| hist_var_cal | 0.001451 ± 0.000583 | 0.0489 ± 0.0152 | 3.7%-7.9% | 5/6 |
| ledoit_wolf | 0.001449 ± 0.000607 | 0.0487 ± 0.0218 | 3.0%-8.8% | 5/6 |
| ledoit_wolf_cal | 0.001445 ± 0.000577 | 0.0475 ± 0.0092 | 3.9%-6.5% | 4/6 |
| lstm | 0.001457 ± 0.000694 | 0.0562 ± 0.0315 | 2.3%-10.7% | 6/6 |
| lstm_cal | 0.001446 ± 0.000644 | 0.0489 ± 0.0196 | 2.3%-7.8% | 5/6 |
| naive | 0.001406 ± 0.000559 | 0.0598 ± 0.0226 | 2.8%-8.8% | 6/6 |
| naive_cal | 0.001423 ± 0.000573 | 0.0490 ± 0.0167 | 3.1%-7.7% | 3/6 |
| param_var | 0.001445 ± 0.000646 | 0.0627 ± 0.0283 | 2.8%-9.5% | 5/6 |
| param_var_cal | 0.001456 ± 0.000650 | 0.0486 ± 0.0212 | 2.5%-8.2% | 6/6 |
| xgboost | 0.001411 ± 0.000607 | 0.0578 ± 0.0229 | 3.0%-8.4% | 6/6 |
| xgboost_cal | 0.001436 ± 0.000628 | 0.0484 ± 0.0191 | 2.6%-7.6% | 6/6 |


Models ending in `_cal` are the same VaR forecasts after the online conformal correction (applied to every VaR model, baselines included). Pinball is shown with six decimals because the models differ in the fourth.

`kupiec_reject` = folds (test years) where the pooled Kupiec test rejects at 5%; `exc_range` = lowest-highest yearly exceedance rate. Pooled tests overstate the sample size (exceedances are dependent across portfolios).

## 95% 1-day expected shortfall

| model | fz0 | es_resid | es_ratio |
|---|---|---|---|
| combo_hg | -3.6507 ± 0.4311 | -0.1257 ± 0.0918 | 1.1262 ± 0.0752 |
| ewma | -3.6507 ± 0.4574 | -0.1305 ± 0.0857 | 1.1266 ± 0.0677 |
| fhs | -3.6494 ± 0.4228 | -0.0279 ± 0.1054 | 1.0253 ± 0.0825 |
| garch | -3.6478 ± 0.4073 | -0.1175 ± 0.0979 | 1.1155 ± 0.0748 |
| gjr | -3.6646 ± 0.3917 | -0.0362 ± 0.1036 | 1.0243 ± 0.0691 |
| har | -3.6285 ± 0.4863 | -0.1412 ± 0.0956 | 1.1426 ± 0.0848 |
| hist_var | -3.6055 ± 0.4863 | 0.0171 ± 0.0320 | 0.9705 ± 0.0519 |
| ledoit_wolf | -3.6115 ± 0.4737 | -0.1412 ± 0.0573 | 1.1408 ± 0.0446 |
| naive | -3.6348 ± 0.4835 | -0.1440 ± 0.1031 | 1.1372 ± 0.0786 |
| param_var | -3.6040 ± 0.4948 | -0.1405 ± 0.0842 | 1.1433 ± 0.0861 |


`fz0` = Fissler-Ziegel/Patton joint (VaR, ES) loss, lower is better and the only ranking score; `es_resid` = mean of (return - ES)/|ES| on the VaR exceedances (0 if right, negative = ES too shallow); `es_ratio` = realised/predicted average exceedance (1 if right). Only the baselines have an ES; the ML models have no ES head. There are only about 2.5 exceedances per portfolio-year, so these checks have low power.

## Primary metric and folds beating the reference

(reference: naive for vol, climatology for drawdown, historical VaR for VaR; lower is better)

| target | metric | model | mean ± std | folds beating reference |
|---|---|---|---|---|
| vol | rmse | combo_hg | 0.0717 ± 0.0493 | 5/6 |
| vol | rmse | combo_hgx | 0.0697 ± 0.0491 | 6/6 |
| vol | rmse | ewma | 0.0781 ± 0.0580 | 6/6 |
| vol | rmse | garch | 0.0748 ± 0.0453 | 5/6 |
| vol | rmse | gjr | 0.0734 ± 0.0416 | 5/6 |
| vol | rmse | har | 0.0728 ± 0.0539 | 6/6 |
| vol | rmse | ledoit_wolf | 0.0834 ± 0.0609 | 3/6 |
| vol | rmse | lstm | 0.0718 ± 0.0458 | 4/6 |
| vol | rmse | naive | 0.0834 ± 0.0590 | 0/6 |
| vol | rmse | rf | 0.0724 ± 0.0532 | 6/6 |
| vol | rmse | xgboost | 0.0710 ± 0.0496 | 6/6 |
| dd | brier | climatology | 0.2199 ± 0.0351 | 0/6 |
| dd | brier | combo_hg | 0.2144 ± 0.0436 | 4/6 |
| dd | brier | combo_hgx | 0.2068 ± 0.0408 | 5/6 |
| dd | brier | ewma | 0.2237 ± 0.0676 | 3/6 |
| dd | brier | fhs_mc | 0.2050 ± 0.0461 | 4/6 |
| dd | brier | fhs_mc_logit | 0.2046 ± 0.0392 | 5/6 |
| dd | brier | garch | 0.2225 ± 0.0278 | 3/6 |
| dd | brier | gjr | 0.2196 ± 0.0280 | 3/6 |
| dd | brier | har | 0.2134 ± 0.0570 | 4/6 |
| dd | brier | ledoit_wolf | 0.2420 ± 0.0886 | 2/6 |
| dd | brier | lstm | 0.2062 ± 0.0464 | 4/6 |
| dd | brier | naive | 0.2229 ± 0.0646 | 2/6 |
| dd | brier | rf | 0.2061 ± 0.0478 | 5/6 |
| dd | brier | xgboost | 0.2101 ± 0.0452 | 5/6 |
| var | pinball | combo_hg | 0.0014 ± 0.0005 | 5/6 |
| var | pinball | combo_hg_cal | 0.0014 ± 0.0005 | 5/6 |
| var | pinball | combo_hgx | 0.0014 ± 0.0005 | 5/6 |
| var | pinball | combo_hgx_cal | 0.0014 ± 0.0006 | 5/6 |
| var | pinball | ewma | 0.0014 ± 0.0006 | 5/6 |
| var | pinball | ewma_cal | 0.0014 ± 0.0006 | 4/6 |
| var | pinball | fhs | 0.0014 ± 0.0005 | 5/6 |
| var | pinball | fhs_cal | 0.0014 ± 0.0005 | 5/6 |
| var | pinball | garch | 0.0014 ± 0.0005 | 4/6 |
| var | pinball | garch_cal | 0.0014 ± 0.0005 | 5/6 |
| var | pinball | gjr | 0.0014 ± 0.0005 | 4/6 |
| var | pinball | gjr_cal | 0.0014 ± 0.0005 | 5/6 |
| var | pinball | har | 0.0014 ± 0.0006 | 5/6 |
| var | pinball | har_cal | 0.0014 ± 0.0006 | 5/6 |
| var | pinball | hist_var | 0.0015 ± 0.0007 | 0/6 |
| var | pinball | hist_var_cal | 0.0015 ± 0.0006 | 0/6 |
| var | pinball | ledoit_wolf | 0.0014 ± 0.0006 | 4/6 |
| var | pinball | ledoit_wolf_cal | 0.0014 ± 0.0006 | 4/6 |
| var | pinball | lstm | 0.0015 ± 0.0007 | 3/6 |
| var | pinball | lstm_cal | 0.0014 ± 0.0006 | 3/6 |
| var | pinball | naive | 0.0014 ± 0.0006 | 5/6 |
| var | pinball | naive_cal | 0.0014 ± 0.0006 | 4/6 |
| var | pinball | param_var | 0.0014 ± 0.0006 | 5/6 |
| var | pinball | param_var_cal | 0.0015 ± 0.0006 | 4/6 |
| var | pinball | xgboost | 0.0014 ± 0.0006 | 5/6 |
| var | pinball | xgboost_cal | 0.0014 ± 0.0006 | 4/6 |


## VaR: share of portfolios rejected by their own Kupiec test (5% level)

| model | kupiec_reject_share |
|---|---|
| combo_hg | 0.01 |
| combo_hg_cal | 0.0225 |
| combo_hgx | 0.0125 |
| combo_hgx_cal | 0.02 |
| ewma | 0.005 |
| ewma_cal | 0.0125 |
| fhs | 0.01 |
| fhs_cal | 0.02 |
| garch | 0.025 |
| garch_cal | 0.0225 |
| gjr | 0.0075 |
| gjr_cal | 0.02 |
| har | 0.0275 |
| har_cal | 0.02 |
| hist_var | 0.015 |
| hist_var_cal | 0.005 |
| ledoit_wolf | 0.0075 |
| ledoit_wolf_cal | 0.01 |
| lstm | 0.0525 |
| lstm_cal | 0.01 |
| naive | 0.045 |
| naive_cal | 0.0175 |
| param_var | 0.0725 |
| param_var_cal | 0.0125 |
| xgboost | 0.0325 |
| xgboost_cal | 0.015 |


## Held-out portfolios (ids never used for training)

### Volatility

| model | rmse | mae | dir_acc | qlike | mse_var |
|---|---|---|---|---|---|
| combo_hg | 0.0728 ± 0.0496 | 0.0491 ± 0.0210 | 0.6599 ± 0.0835 | 0.3232 ± 0.3295 | 0.005946 ± 0.013283 |
| combo_hgx | 0.0708 ± 0.0495 | 0.0472 ± 0.0208 | 0.6636 ± 0.0846 | 0.3053 ± 0.3109 | 0.005863 ± 0.013145 |
| ewma | 0.0794 ± 0.0587 | 0.0559 ± 0.0378 | 0.6151 ± 0.0674 | 0.3539 ± 0.3585 | 0.007465 ± 0.016636 |
| garch | 0.0760 ± 0.0457 | 0.0529 ± 0.0180 | 0.6598 ± 0.0906 | 0.3178 ± 0.2748 | 0.005921 ± 0.013045 |
| gjr | 0.0746 ± 0.0419 | 0.0523 ± 0.0166 | 0.6656 ± 0.0966 | 0.2883 ± 0.2048 | 0.005522 ± 0.012011 |
| har | 0.0738 ± 0.0544 | 0.0489 ± 0.0259 | 0.6486 ± 0.0766 | 0.3694 ± 0.4183 | 0.006360 ± 0.014270 |
| ledoit_wolf | 0.0846 ± 0.0615 | 0.0627 ± 0.0415 | 0.6280 ± 0.0517 | 0.4711 ± 0.5396 | 0.007091 ± 0.015691 |
| lstm | 0.0727 ± 0.0462 | 0.0486 ± 0.0217 | 0.6768 ± 0.0670 | 0.3046 ± 0.2702 | 0.005729 ± 0.012557 |
| naive | 0.0848 ± 0.0598 | 0.0582 ± 0.0350 | nan ± nan | 0.4088 ± 0.3457 | 0.009231 ± 0.020562 |
| rf | 0.0734 ± 0.0538 | 0.0486 ± 0.0247 | 0.6560 ± 0.0653 | 0.3469 ± 0.3779 | 0.006468 ± 0.014523 |
| xgboost | 0.0719 ± 0.0500 | 0.0482 ± 0.0233 | 0.6642 ± 0.0613 | 0.3122 ± 0.2966 | 0.006056 ± 0.013534 |


### Drawdown

| model | auc | brier | logloss |
|---|---|---|---|
| climatology | 0.5000 ± 0.0000 | 0.2174 ± 0.0368 | 0.6267 ± 0.0759 |
| combo_hg | 0.6208 ± 0.0665 | 0.2125 ± 0.0463 | 0.6134 ± 0.1005 |
| combo_hgx | 0.6310 ± 0.0552 | 0.2047 ± 0.0439 | 0.5965 ± 0.0950 |
| ewma | 0.6071 ± 0.0747 | 0.2224 ± 0.0713 | 0.6422 ± 0.1790 |
| fhs_mc | 0.6232 ± 0.0529 | 0.2023 ± 0.0496 | 0.5920 ± 0.1117 |
| fhs_mc_logit | 0.6232 ± 0.0529 | 0.2026 ± 0.0420 | 0.5932 ± 0.0902 |
| garch | 0.6265 ± 0.0558 | 0.2209 ± 0.0299 | 0.6341 ± 0.0646 |
| gjr | 0.6398 ± 0.0611 | 0.2179 ± 0.0304 | 0.6284 ± 0.0659 |
| har | 0.6057 ± 0.0745 | 0.2112 ± 0.0603 | 0.6082 ± 0.1361 |
| ledoit_wolf | 0.5797 ± 0.1387 | 0.2407 ± 0.0913 | 0.6815 ± 0.2182 |
| lstm | 0.6305 ± 0.0626 | 0.2036 ± 0.0492 | 0.5954 ± 0.1080 |
| naive | 0.6094 ± 0.0884 | 0.2215 ± 0.0690 | 0.6458 ± 0.1734 |
| rf | 0.6043 ± 0.0357 | 0.2035 ± 0.0513 | 0.5935 ± 0.1140 |
| xgboost | 0.6157 ± 0.0353 | 0.2070 ± 0.0481 | 0.6010 ± 0.1035 |


### VaR

| model | pinball | exceedance | exc_range | kupiec_reject |
|---|---|---|---|---|
| combo_hg | 0.001397 ± 0.000543 | 0.0498 ± 0.0266 | 2.3%-8.5% | 5/6 |
| combo_hg_cal | 0.001399 ± 0.000558 | 0.0471 ± 0.0165 | 3.1%-7.6% | 4/6 |
| combo_hgx | 0.001388 ± 0.000566 | 0.0493 ± 0.0230 | 2.6%-8.2% | 5/6 |
| combo_hgx_cal | 0.001399 ± 0.000587 | 0.0465 ± 0.0152 | 3.2%-7.4% | 5/6 |
| ewma | 0.001403 ± 0.000573 | 0.0540 ± 0.0220 | 2.7%-8.0% | 5/6 |
| ewma_cal | 0.001422 ± 0.000594 | 0.0488 ± 0.0145 | 3.3%-7.3% | 5/6 |
| fhs | 0.001396 ± 0.000529 | 0.0524 ± 0.0266 | 2.7%-8.7% | 5/6 |
| fhs_cal | 0.001401 ± 0.000538 | 0.0479 ± 0.0165 | 3.0%-7.6% | 4/6 |
| garch | 0.001394 ± 0.000502 | 0.0462 ± 0.0267 | 2.1%-8.2% | 5/6 |
| garch_cal | 0.001396 ± 0.000528 | 0.0468 ± 0.0155 | 2.9%-7.4% | 4/6 |
| gjr | 0.001364 ± 0.000467 | 0.0462 ± 0.0256 | 2.0%-8.0% | 5/6 |
| gjr_cal | 0.001367 ± 0.000491 | 0.0465 ± 0.0146 | 3.0%-7.1% | 4/6 |
| har | 0.001421 ± 0.000617 | 0.0549 ± 0.0284 | 2.6%-9.5% | 5/6 |
| har_cal | 0.001417 ± 0.000606 | 0.0484 ± 0.0163 | 3.4%-7.8% | 4/6 |
| hist_var | 0.001482 ± 0.000702 | 0.0583 ± 0.0275 | 3.6%-10.9% | 5/6 |
| hist_var_cal | 0.001455 ± 0.000605 | 0.0497 ± 0.0144 | 3.8%-7.7% | 4/6 |
| ledoit_wolf | 0.001452 ± 0.000632 | 0.0487 ± 0.0215 | 3.1%-8.7% | 4/6 |
| ledoit_wolf_cal | 0.001449 ± 0.000602 | 0.0478 ± 0.0082 | 3.8%-6.3% | 2/6 |
| lstm | 0.001458 ± 0.000720 | 0.0551 ± 0.0320 | 2.4%-10.8% | 6/6 |
| lstm_cal | 0.001445 ± 0.000672 | 0.0480 ± 0.0178 | 2.6%-7.7% | 5/6 |
| naive | 0.001409 ± 0.000582 | 0.0605 ± 0.0236 | 2.9%-9.0% | 5/6 |
| naive_cal | 0.001425 ± 0.000593 | 0.0494 ± 0.0161 | 3.2%-7.8% | 3/6 |
| param_var | 0.001453 ± 0.000673 | 0.0636 ± 0.0279 | 3.4%-9.6% | 5/6 |
| param_var_cal | 0.001460 ± 0.000672 | 0.0489 ± 0.0205 | 2.8%-8.2% | 5/6 |
| xgboost | 0.001413 ± 0.000623 | 0.0579 ± 0.0218 | 3.5%-8.4% | 6/6 |
| xgboost_cal | 0.001437 ± 0.000642 | 0.0486 ± 0.0180 | 2.7%-7.4% | 5/6 |


### Expected shortfall

| model | fz0 | es_resid | es_ratio |
|---|---|---|---|
| combo_hg | -3.6478 ± 0.4519 | -0.1290 ± 0.1036 | 1.1308 ± 0.0857 |
| ewma | -3.6501 ± 0.4765 | -0.1433 ± 0.0938 | 1.1394 ± 0.0736 |
| fhs | -3.6481 ± 0.4396 | -0.0345 ± 0.1205 | 1.0310 ± 0.0962 |
| garch | -3.6462 ± 0.4240 | -0.1251 ± 0.1047 | 1.1248 ± 0.0809 |
| gjr | -3.6678 ± 0.4041 | -0.0439 ± 0.1124 | 1.0320 ± 0.0753 |
| har | -3.6248 ± 0.5124 | -0.1477 ± 0.1055 | 1.1501 ± 0.0931 |
| hist_var | -3.6036 ± 0.5061 | 0.0159 ± 0.0321 | 0.9725 ± 0.0479 |
| ledoit_wolf | -3.6088 ± 0.4968 | -0.1468 ± 0.0721 | 1.1461 ± 0.0568 |
| naive | -3.6351 ± 0.5093 | -0.1473 ± 0.1092 | 1.1428 ± 0.0812 |
| param_var | -3.5992 ± 0.5159 | -0.1437 ± 0.0904 | 1.1479 ± 0.0928 |


## Drawdown calibration: mean predicted probability vs realized rate

| model | fold | predicted | realized |
|---|---|---|---|
| climatology | 2020 | 0.344 | 0.45100000500679016 |
| climatology | 2021 | 0.353 | 0.34200000762939453 |
| climatology | 2022 | 0.352 | 0.43299999833106995 |
| climatology | 2023 | 0.359 | 0.11999999731779099 |
| climatology | 2024 | 0.343 | 0.31700000166893005 |
| climatology | 2025 | 0.337 | 0.27399998903274536 |
| combo_hg | 2020 | 0.578 | 0.45100000500679016 |
| combo_hg | 2021 | 0.441 | 0.34200000762939453 |
| combo_hg | 2022 | 0.456 | 0.43299999833106995 |
| combo_hg | 2023 | 0.311 | 0.11999999731779099 |
| combo_hg | 2024 | 0.384 | 0.31700000166893005 |
| combo_hg | 2025 | 0.37 | 0.27399998903274536 |
| combo_hgx | 2020 | 0.504 | 0.45100000500679016 |
| combo_hgx | 2021 | 0.415 | 0.34200000762939453 |
| combo_hgx | 2022 | 0.426 | 0.43299999833106995 |
| combo_hgx | 2023 | 0.3 | 0.11999999731779099 |
| combo_hgx | 2024 | 0.361 | 0.31700000166893005 |
| combo_hgx | 2025 | 0.356 | 0.27399998903274536 |
| ewma | 2020 | 0.621 | 0.45100000500679016 |
| ewma | 2021 | 0.422 | 0.34200000762939453 |
| ewma | 2022 | 0.466 | 0.43299999833106995 |
| ewma | 2023 | 0.193 | 0.11999999731779099 |
| ewma | 2024 | 0.362 | 0.31700000166893005 |
| ewma | 2025 | 0.321 | 0.27399998903274536 |
| fhs_mc | 2020 | 0.439 | 0.45100000500679016 |
| fhs_mc | 2021 | 0.336 | 0.34200000762939453 |
| fhs_mc | 2022 | 0.343 | 0.43299999833106995 |
| fhs_mc | 2023 | 0.258 | 0.11999999731779099 |
| fhs_mc | 2024 | 0.287 | 0.31700000166893005 |
| fhs_mc | 2025 | 0.288 | 0.27399998903274536 |
| fhs_mc_logit | 2020 | 0.415 | 0.45100000500679016 |
| fhs_mc_logit | 2021 | 0.349 | 0.34200000762939453 |
| fhs_mc_logit | 2022 | 0.352 | 0.43299999833106995 |
| fhs_mc_logit | 2023 | 0.302 | 0.11999999731779099 |
| fhs_mc_logit | 2024 | 0.308 | 0.31700000166893005 |
| fhs_mc_logit | 2025 | 0.306 | 0.27399998903274536 |
| garch | 2020 | 0.561 | 0.45100000500679016 |
| garch | 2021 | 0.469 | 0.34200000762939453 |
| garch | 2022 | 0.483 | 0.43299999833106995 |
| garch | 2023 | 0.382 | 0.11999999731779099 |
| garch | 2024 | 0.426 | 0.31700000166893005 |
| garch | 2025 | 0.416 | 0.27399998903274536 |
| gjr | 2020 | 0.552 | 0.45100000500679016 |
| gjr | 2021 | 0.458 | 0.34200000762939453 |
| gjr | 2022 | 0.48 | 0.43299999833106995 |
| gjr | 2023 | 0.38 | 0.11999999731779099 |
| gjr | 2024 | 0.428 | 0.31700000166893005 |
| gjr | 2025 | 0.419 | 0.27399998903274536 |
| har | 2020 | 0.595 | 0.45100000500679016 |
| har | 2021 | 0.412 | 0.34200000762939453 |
| har | 2022 | 0.43 | 0.43299999833106995 |
| har | 2023 | 0.239 | 0.11999999731779099 |
| har | 2024 | 0.342 | 0.31700000166893005 |
| har | 2025 | 0.323 | 0.27399998903274536 |
| ledoit_wolf | 2020 | 0.719 | 0.45100000500679016 |
| ledoit_wolf | 2021 | 0.574 | 0.34200000762939453 |
| ledoit_wolf | 2022 | 0.47 | 0.43299999833106995 |
| ledoit_wolf | 2023 | 0.298 | 0.11999999731779099 |
| ledoit_wolf | 2024 | 0.281 | 0.31700000166893005 |
| ledoit_wolf | 2025 | 0.338 | 0.27399998903274536 |
| lstm | 2020 | 0.442 | 0.45100000500679016 |
| lstm | 2021 | 0.328 | 0.34200000762939453 |
| lstm | 2022 | 0.367 | 0.43299999833106995 |
| lstm | 2023 | 0.277 | 0.11999999731779099 |
| lstm | 2024 | 0.309 | 0.31700000166893005 |
| lstm | 2025 | 0.29 | 0.27399998903274536 |
| naive | 2020 | 0.569 | 0.45100000500679016 |
| naive | 2021 | 0.414 | 0.34200000762939453 |
| naive | 2022 | 0.438 | 0.43299999833106995 |
| naive | 2023 | 0.185 | 0.11999999731779099 |
| naive | 2024 | 0.345 | 0.31700000166893005 |
| naive | 2025 | 0.307 | 0.27399998903274536 |
| rf | 2020 | 0.438 | 0.45100000500679016 |
| rf | 2021 | 0.372 | 0.34200000762939453 |
| rf | 2022 | 0.372 | 0.43299999833106995 |
| rf | 2023 | 0.241 | 0.11999999731779099 |
| rf | 2024 | 0.313 | 0.31700000166893005 |
| rf | 2025 | 0.3 | 0.27399998903274536 |
| xgboost | 2020 | 0.356 | 0.45100000500679016 |
| xgboost | 2021 | 0.365 | 0.34200000762939453 |
| xgboost | 2022 | 0.367 | 0.43299999833106995 |
| xgboost | 2023 | 0.278 | 0.11999999731779099 |
| xgboost | 2024 | 0.314 | 0.31700000166893005 |
| xgboost | 2025 | 0.329 | 0.27399998903274536 |


## Diebold-Mariano tests (date-level, HAC variance)

Negative mean loss difference = the ML model has the lower loss. `significant` is p < 0.05. ~330 dates, so low power.

| target | model | vs | baseline | mean_loss_diff | dm_stat | p | better | significant | n_dates |
|---|---|---|---|---|---|---|---|---|---|
| vol | rf | best_baseline | gjr | 0.0006281 | 0.6174 | 0.5374 | False | False | 333 |
| vol | rf | reference | naive | -0.00212 | -1.716 | 0.08709 | True | False | 333 |
| vol | xgboost | best_baseline | gjr | 0.0002331 | 0.3547 | 0.7231 | False | False | 333 |
| vol | xgboost | reference | naive | -0.002515 | -1.706 | 0.08891 | True | False | 333 |
| vol | lstm | best_baseline | gjr | 9.018e-05 | 0.2018 | 0.8402 | False | False | 333 |
| vol | lstm | reference | naive | -0.002658 | -1.571 | 0.1171 | True | False | 333 |
| dd | rf | best_baseline | fhs_mc_logit | 0.001701 | 0.3828 | 0.7021 | False | False | 333 |
| dd | rf | reference | climatology | -0.01259 | -1.806 | 0.07178 | True | False | 333 |
| dd | xgboost | best_baseline | fhs_mc_logit | 0.006141 | 0.8074 | 0.42 | False | False | 333 |
| dd | xgboost | reference | climatology | -0.008153 | -1.291 | 0.1976 | True | False | 333 |
| dd | lstm | best_baseline | fhs_mc_logit | 0.001655 | 0.4274 | 0.6694 | False | False | 333 |
| dd | lstm | reference | climatology | -0.01264 | -1.703 | 0.08956 | True | False | 333 |
| var | xgboost | best_baseline | gjr | 3.668e-05 | 0.9524 | 0.3416 | False | False | 333 |
| var | xgboost | reference | hist_var | -6.231e-05 | -0.769 | 0.4424 | True | False | 333 |
| var | lstm | best_baseline | gjr | 8.028e-05 | 0.7322 | 0.4646 | False | False | 333 |
| var | lstm | reference | hist_var | -1.871e-05 | -0.442 | 0.6588 | True | False | 333 |
| var | xgboost_cal | best_baseline | gjr_cal | 5.829e-05 | 1.481 | 0.1396 | False | False | 333 |
| var | xgboost_cal | reference | hist_var_cal | -1.506e-05 | -0.2368 | 0.813 | True | False | 333 |
| var | lstm_cal | best_baseline | gjr_cal | 7.048e-05 | 0.8641 | 0.3881 | False | False | 333 |
| var | lstm_cal | reference | hist_var_cal | -2.864e-06 | -0.07452 | 0.9406 | True | False | 333 |


## Model confidence set (90%, Hansen-Lunde-Nason)

Models that cannot be statistically separated from the best on date-level loss.

| target | in the 90% confidence set | eliminated |
|---|---|---|
| vol | ledoit_wolf, naive, ewma, rf, garch, har, combo_hg, xgboost, lstm, gjr, combo_hgx |  |
| dd | ledoit_wolf, garch, ewma, naive, gjr, climatology, combo_hg, har, xgboost, combo_hgx, lstm, rf, fhs_mc, fhs_mc_logit |  |
| var | param_var_cal, param_var, hist_var_cal, hist_var, xgboost_cal, ledoit_wolf_cal, naive_cal, lstm_cal, ledoit_wolf, ewma_cal, har_cal, lstm, naive, xgboost, fhs_cal, har, ewma, combo_hg_cal, garch, garch_cal, fhs, combo_hgx_cal, combo_hg, combo_hgx, gjr_cal, gjr |  |
| es | param_var, hist_var, har, ledoit_wolf, naive, garch, fhs, combo_hg, ewma, gjr |  |


## VaR coverage, date-clustered (HAC) test

The pooled Kupiec test treats ~20,000 rows per year as independent; this one averages the exceedance rate over portfolios per date and tests the ~330 date-level rates.

| model | n_dates | exceed_rate | stat | p | years rejected (p<0.05) |
|---|---|---|---|---|---|
| naive | 333 | 0.05811 | 1.005 | 0.3158 | 1/7 |
| ewma | 333 | 0.05283 | 0.3533 | 0.7241 | 1/7 |
| ledoit_wolf | 333 | 0.04733 | -0.3118 | 0.7554 | 1/7 |
| har | 333 | 0.0532 | 0.369 | 0.7123 | 1/7 |
| hist_var | 333 | 0.0566 | 0.677 | 0.4989 | 0/7 |
| param_var | 333 | 0.06018 | 1.107 | 0.2692 | 1/7 |
| garch | 333 | 0.04453 | -0.7095 | 0.4785 | 2/7 |
| fhs | 333 | 0.05101 | 0.1229 | 0.9022 | 1/7 |
| gjr | 333 | 0.04543 | -0.6084 | 0.5433 | 2/7 |
| xgboost | 333 | 0.05658 | 0.8069 | 0.4203 | 1/7 |
| lstm | 333 | 0.05676 | 0.7358 | 0.4624 | 1/7 |
| combo_hg | 333 | 0.04768 | -0.2887 | 0.773 | 1/7 |
| combo_hgx | 333 | 0.04777 | -0.2879 | 0.7736 | 1/7 |
| naive_cal | 333 | 0.04908 | -0.1286 | 0.8978 | 0/7 |
| ewma_cal | 333 | 0.04821 | -0.2469 | 0.8051 | 0/7 |
| ledoit_wolf_cal | 333 | 0.04764 | -0.3058 | 0.76 | 0/7 |
| har_cal | 333 | 0.04784 | -0.2861 | 0.7749 | 0/7 |
| hist_var_cal | 333 | 0.04873 | -0.1612 | 0.8721 | 0/7 |
| param_var_cal | 333 | 0.04835 | -0.2172 | 0.8282 | 1/7 |
| garch_cal | 333 | 0.04619 | -0.5248 | 0.6001 | 1/7 |
| fhs_cal | 333 | 0.04729 | -0.3681 | 0.713 | 1/7 |
| gjr_cal | 333 | 0.04622 | -0.534 | 0.5937 | 1/7 |
| xgboost_cal | 333 | 0.04797 | -0.2792 | 0.7803 | 1/7 |
| lstm_cal | 333 | 0.05049 | 0.06009 | 0.9521 | 1/7 |
| combo_hg_cal | 333 | 0.04678 | -0.4371 | 0.6623 | 1/7 |
| combo_hgx_cal | 333 | 0.04643 | -0.5029 | 0.6154 | 0/7 |


## Conformal calibration: average exceedance and drift bound

`mean_err` is the average share of portfolios below the calibrated VaR (target 0.05); by the tracker identity mean_err - 0.05 = `drift_bound` exactly. A small number says the long-run average is calibrated, nothing about single years.

| model | mean_err | drift_bound | n_dates |
|---|---|---|---|
| combo_hg | 0.0468 | -0.0032 | 333 |
| combo_hgx | 0.0466 | -0.0034 | 333 |
| ewma | 0.0482 | -0.0018 | 333 |
| fhs | 0.0473 | -0.0027 | 333 |
| garch | 0.0461 | -0.0039 | 333 |
| gjr | 0.0462 | -0.0038 | 333 |
| har | 0.0477 | -0.0023 | 333 |
| hist_var | 0.0486 | -0.0014 | 333 |
| ledoit_wolf | 0.0476 | -0.0024 | 333 |
| lstm | 0.0508 | 0.0008 | 333 |
| naive | 0.0491 | -0.0009 | 333 |
| param_var | 0.0484 | -0.0016 | 333 |
| xgboost | 0.048 | -0.002 | 333 |


## Step-size sensitivity (reported only; eta = 0.25 was fixed beforehand)

| model | eta | exceedance | exc_range | kupiec_reject |
|---|---|---|---|---|
| combo_hg_cal | 0.1 | 0.0451 ± 0.0210 | 2.6%-7.9% | 6/6 |
| combo_hg_cal | 0.25 | 0.0466 ± 0.0164 | 3.1%-7.6% | 5/6 |
| combo_hg_cal | 0.5 | 0.0484 ± 0.0136 | 3.0%-7.2% | 5/6 |
| combo_hgx_cal | 0.1 | 0.0456 ± 0.0191 | 2.6%-7.8% | 6/6 |
| combo_hgx_cal | 0.25 | 0.0466 ± 0.0152 | 3.1%-7.5% | 5/6 |
| combo_hgx_cal | 0.5 | 0.0482 ± 0.0139 | 2.8%-7.1% | 4/6 |
| ewma_cal | 0.1 | 0.0481 ± 0.0184 | 2.6%-7.7% | 6/6 |
| ewma_cal | 0.25 | 0.0483 ± 0.0152 | 3.2%-7.4% | 5/6 |
| ewma_cal | 0.5 | 0.0490 ± 0.0139 | 2.9%-7.0% | 5/6 |
| fhs_cal | 0.1 | 0.0466 ± 0.0223 | 2.6%-8.2% | 6/6 |
| fhs_cal | 0.25 | 0.0473 ± 0.0169 | 3.0%-7.7% | 6/6 |
| fhs_cal | 0.5 | 0.0488 ± 0.0141 | 3.2%-7.3% | 5/6 |
| garch_cal | 0.1 | 0.0437 ± 0.0205 | 2.4%-7.7% | 6/6 |
| garch_cal | 0.25 | 0.0460 ± 0.0158 | 2.8%-7.3% | 5/6 |
| garch_cal | 0.5 | 0.0481 ± 0.0135 | 3.2%-7.0% | 6/6 |
| gjr_cal | 0.1 | 0.0439 ± 0.0199 | 2.4%-7.6% | 6/6 |
| gjr_cal | 0.25 | 0.0460 ± 0.0152 | 2.9%-7.2% | 4/6 |
| gjr_cal | 0.5 | 0.0481 ± 0.0130 | 3.2%-7.0% | 5/6 |
| har_cal | 0.1 | 0.0471 ± 0.0229 | 2.5%-8.6% | 6/6 |
| har_cal | 0.25 | 0.0477 ± 0.0172 | 3.2%-7.9% | 4/6 |
| har_cal | 0.5 | 0.0490 ± 0.0149 | 2.9%-7.5% | 4/6 |
| hist_var_cal | 0.1 | 0.0491 ± 0.0227 | 3.2%-9.4% | 5/6 |
| hist_var_cal | 0.25 | 0.0489 ± 0.0152 | 3.7%-7.9% | 5/6 |
| hist_var_cal | 0.5 | 0.0498 ± 0.0116 | 3.3%-6.9% | 3/6 |
| ledoit_wolf_cal | 0.1 | 0.0460 ± 0.0154 | 3.2%-7.5% | 6/6 |
| ledoit_wolf_cal | 0.25 | 0.0475 ± 0.0092 | 3.9%-6.5% | 4/6 |
| ledoit_wolf_cal | 0.5 | 0.0491 ± 0.0073 | 3.8%-6.1% | 2/6 |
| lstm_cal | 0.1 | 0.0490 ± 0.0253 | 2.6%-9.0% | 6/6 |
| lstm_cal | 0.25 | 0.0489 ± 0.0196 | 2.3%-7.8% | 5/6 |
| lstm_cal | 0.5 | 0.0495 ± 0.0163 | 2.2%-7.1% | 5/6 |
| naive_cal | 0.1 | 0.0499 ± 0.0203 | 2.5%-8.3% | 5/6 |
| naive_cal | 0.25 | 0.0490 ± 0.0167 | 3.1%-7.7% | 3/6 |
| naive_cal | 0.5 | 0.0495 ± 0.0149 | 2.8%-7.3% | 5/6 |
| param_var_cal | 0.1 | 0.0504 ± 0.0250 | 2.0%-8.9% | 6/6 |
| param_var_cal | 0.25 | 0.0486 ± 0.0212 | 2.5%-8.2% | 6/6 |
| param_var_cal | 0.5 | 0.0488 ± 0.0185 | 2.5%-7.6% | 5/6 |
| xgboost_cal | 0.1 | 0.0499 ± 0.0210 | 2.6%-8.0% | 5/6 |
| xgboost_cal | 0.25 | 0.0484 ± 0.0191 | 2.6%-7.6% | 6/6 |
| xgboost_cal | 0.5 | 0.0485 ± 0.0174 | 2.5%-7.1% | 6/6 |


## Training diagnostics per fold

XGBoost rounds = argmin of each head's validation curve (`xgb_at_cap` lists heads that hit the 600-round cap); LSTM epochs = each head's best validation epoch.

| fold | xgb_reg | xgb_clf | xgb_qr | xgb_at_cap | lstm_vol | lstm_dd | lstm_var |
|---|---|---|---|---|---|---|---|
| 2020 | 599 | 5 | 182 |  | 4 | 1 | 14 |
| 2021 | 595 | 27 | 24 |  | 3 | 5 | 2 |
| 2022 | 599 | 17 | 155 |  | 4 | 4 | 1 |
| 2023 | 110 | 35 | 595 |  | 2 | 1 | 2 |
| 2024 | 511 | 82 | 599 |  | 1 | 3 | 3 |
| 2025 | 600 | 565 | 600 | reg,qr | 5 | 2 | 12 |


## Seed spread, Round 3 ML configuration (6-year means across seeds)

| target | model | metric | mean | std | min | max |
|---|---|---|---|---|---|---|
| dd | lstm | auc | 0.61584 | 0.01079 | 0.60821 | 0.62346 |
| dd | lstm | brier | 0.20833 | 0.00077 | 0.20779 | 0.20888 |
| dd | lstm | logloss | 0.60472 | 0.00259 | 0.60289 | 0.60655 |
| dd | rf | auc | 0.58211 | 0.00178 | 0.58086 | 0.58337 |
| dd | rf | brier | 0.20823 | 0.00009 | 0.20816 | 0.20829 |
| dd | rf | logloss | 0.60362 | 0.00021 | 0.60347 | 0.60377 |
| dd | xgboost | auc | 0.58242 | 0.01581 | 0.57124 | 0.59360 |
| dd | xgboost | brier | 0.21148 | 0.00164 | 0.21032 | 0.21263 |
| dd | xgboost | logloss | 0.61121 | 0.00368 | 0.60861 | 0.61382 |
| var | lstm | exceedance | 0.05365 | 0.00110 | 0.05287 | 0.05442 |
| var | lstm | pinball | 0.00138 | 0.00004 | 0.00135 | 0.00141 |
| var | lstm_cal | exceedance | 0.04703 | 0.00007 | 0.04698 | 0.04709 |
| var | lstm_cal | pinball | 0.00138 | 0.00004 | 0.00135 | 0.00140 |
| var | xgboost | exceedance | 0.05713 | 0.00139 | 0.05614 | 0.05811 |
| var | xgboost | pinball | 0.00141 | 0.00001 | 0.00141 | 0.00141 |
| var | xgboost_cal | exceedance | 0.04827 | 0.00007 | 0.04823 | 0.04832 |
| var | xgboost_cal | pinball | 0.00144 | 0.00001 | 0.00143 | 0.00144 |
| vol | lstm | dir_acc | 0.66632 | 0.01092 | 0.65859 | 0.67404 |
| vol | lstm | mae | 0.04803 | 0.00008 | 0.04797 | 0.04809 |
| vol | lstm | rmse | 0.07106 | 0.00124 | 0.07019 | 0.07193 |
| vol | rf | dir_acc | 0.65870 | 0.00110 | 0.65792 | 0.65948 |
| vol | rf | mae | 0.04846 | 0.00002 | 0.04844 | 0.04848 |
| vol | rf | rmse | 0.07250 | 0.00009 | 0.07244 | 0.07257 |
| vol | xgboost | dir_acc | 0.66892 | 0.00308 | 0.66674 | 0.67110 |
| vol | xgboost | mae | 0.04864 | 0.00050 | 0.04829 | 0.04899 |
| vol | xgboost | rmse | 0.07232 | 0.00070 | 0.07183 | 0.07282 |


## Random Forest volatility model: top features (impurity)

| feature | importance |
|---|---|
| vol_21 | 0.4359 |
| mkt_vol_21 | 0.1504 |
| har_vol | 0.1031 |
| VIX | 0.0555 |
| FEDFUNDS_d126 | 0.0514 |
| ewma_vol | 0.039 |
| mkt_mom_21 | 0.0226 |
| sortino_21 | 0.0161 |
| beta_mkt_21 | 0.0137 |
| mean_pair_corr_21 | 0.0128 |
| garch_vol21 | 0.0106 |
| sharpe_21 | 0.0104 |
| kurt_63 | 0.0101 |
| vol_126 | 0.0076 |
| corr_mkt_21 | 0.0074 |


## XGBoost volatility model: top features (gain)

| feature | gain |
|---|---|
| har_vol | 0.1067461222410202 |
| vol_21 | 0.09612550586462021 |
| ewma_vol | 0.0833454430103302 |
| mkt_vol_21 | 0.05493659898638725 |
| mean_pair_corr_21 | 0.052338480949401855 |
| beta_mkt_21 | 0.04099678248167038 |
| sharpe_21 | 0.03478848934173584 |
| corr_mkt_21 | 0.03305336460471153 |
| FEDFUNDS_d126 | 0.028550175949931145 |
| DGS10_d63 | 0.02808300033211708 |
| VIX | 0.02726965770125389 |
| DEXINUS_r63 | 0.026710424572229385 |
| DCOILBRENTEU_r63 | 0.026458611711859703 |
| mkt_mom_21 | 0.02611149102449417 |
| sortino_21 | 0.026058219373226166 |


