

UNITTEST()

PEERDIR(
    catboost/libs/metrics
    catboost/libs/algo
)

SRCS(
    accuracy_ut.cpp
    brier_score_ut.cpp
    balanced_accuracy_ut.cpp
    dcg_ut.cpp
    f1_metric_ut.cpp
    hamming_loss_ut.cpp
    hinge_loss_ut.cpp
    kappa_ut.cpp
    llp_ut.cpp
    mcc_metric_ut.cpp
    median_absolute_error_ut.cpp
    msle_ut.cpp
    precision_recall_at_k_ut.cpp
    precision_recall_ut.cpp
    smape_ut.cpp
    zero_one_loss_ut.cpp
)

END()
