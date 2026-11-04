# Formula-to-code mapping

This implementation follows the uploaded report's uncentred residual formulation.
The earlier PPT includes a mean-centred residual variant; these are different
parameterisations. This package consistently uses the report variant below.

| Report equation | Implemented calculation | Location |
|---|---|---|
| 3.1 persistence | bHb=Hb0; bW=W0 (training median if missing); bS=observed five-day mean | data.Preprocessor.references |
| 3.2 standardisation | x'=(filled x - training mean)/training SD; zero SD replaced by 1 | data.Preprocessor.fit/transform |
| 3.3 residual scale | sigma=max(training SD(y-b), floor); floors .1,.1,100 | data.Preprocessor.fit |
| 3.4 residual | r=(y-b)/sigma | data.Preprocessor.residuals |
| 3.5 shared representation | z=tanh(x' W1+b1), 24 hidden units | model.JointMLP |
| 3.6 prediction | yhat=b+sigma*(z W2+b2); steps clipped at zero | model.predict_network |
| 3.7 participant weight | omega=1/number of that person's training anchors | model.fit_network |
| 3.8 masked weighted L1 | sum(omega*m*abs(r-rhat))/sum(omega*m) per task | model.JointMLP.loss_grad |
| 3.9 joint loss | L=L_Hb+alphaW*L_W+alphaS*L_S | model.JointMLP.loss_grad |
| 3.10 candidate selection | argmin mean grouped validation participant-MAE | training.train_all |
| 3.11 acceptance gate | mean(eHbOnly-eBest)>sampleSD(eHbOnly-eBest)/sqrt(K) | model.gate |
| 3.12 MAE | average absolute error per person; average over people | training.participant_errors/metrics |
| 3.13 skill | 1 - MAEmodel/MAEpersistence; undefined when baseline MAE=0 | training.metrics |
| Paired comparison | delta=MAEHbOnly-MAEselected; positive is improvement | training.paired |
| 3.15 cluster conformal | person max absolute calibration error; ranked 90% quantile; yhat +/- q | training.calibrate |

An entirely unobserved task has zero loss and zero gradient; denominator is clamped
to 1e-12 only to avoid division by zero. Masked target entries never enter the loss.
Training uses full-batch Adam (beta1=.9, beta2=.999, epsilon=1e-8), lr=.008.
Epoch count and seed are entered manually. No early stopping or fold-test tuning.

Chapter 3 and Chapter 5 of the supplied report use opposite signs for paired
differences. This application consistently defines positive as improvement.
It uses participant-maximum cluster conformal scores rather than the conflicting
individual-anchor procedure described elsewhere in the report.
