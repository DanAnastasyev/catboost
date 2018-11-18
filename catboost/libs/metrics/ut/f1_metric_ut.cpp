#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <library/unittest/registar.h>

Y_UNIT_TEST_SUITE(F1MetricsTest) {

Y_UNIT_TEST(BinF1Test) {
    {
        TVector <TVector<double>> approx{{0, 1, 0, 0, 1, 0}};
        TVector<float> target{0, 1, 0, 0, 0, 1};
        TVector<float> weight{1, 1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeBinClassF1Metric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.5, 1e-2);
    }
    {
        TVector <TVector<double>> approx{{0, 0, 1}};
        TVector<float> target{0, 1, 1};
        TVector<float> weight{1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeBinClassF1Metric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);;

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.667, 1e-3);
    }
    {
        TVector <TVector<double>> approx{{1, 1, 1, 0}};
        TVector<float> target{1, 1, 1, 0};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeBinClassF1Metric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
    }
    {
        TVector <TVector<double>> approx{{1, 1, 1, 1}};
        TVector<float> target{1, 1, 1, 1};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeBinClassF1Metric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
    }
    {
        TVector <TVector<double>> approx{{0, 0, 0, 0}};
        TVector<float> target{0, 0, 0, 0};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeBinClassF1Metric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0, 1e-1);
    }

}

Y_UNIT_TEST(MulticlassF1Test) {
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0, 0},
                                        {1, 1, 0, 0, 0},
                                        {0, 0, 1, 1, 1}};
        TVector<float> target{0, 1, 2, 1, 0};
        TVector<float> weight{1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeMultiClassF1Metric(1);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.5, 1e-3);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0, 0},
                                        {1, 1, 0, 1, 0},
                                        {0, 0, 1, 0, 1}};
        TVector<float> target{0, 1, 2, 1, 0};
        TVector<float> weight{1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeMultiClassF1Metric(1);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.8, 1e-2);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0, 0},
                                        {1, 1, 1, 1, 1},
                                        {0, 0, 0, 0, 0}};
        TVector<float> target{0, 1, 2, 1, 0};
        TVector<float> weight{1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeMultiClassF1Metric(1);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.571, 1e-3);
    }
}

Y_UNIT_TEST(TotalF1Test) {
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0, 0},
                                        {1, 1, 0, 0, 0},
                                        {0, 0, 1, 1, 1}};
        TVector<float> target{0, 1, 2, 1, 0};
        TVector<float> weight{1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeTotalF1Metric(3);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.3, 1e-2);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0, 0},
                                        {1, 1, 0, 1, 0},
                                        {0, 0, 1, 0, 1}};
        TVector<float> target{0, 1, 2, 1, 0};
        TVector<float> weight{1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeTotalF1Metric(3);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.453, 1e-3);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0, 0},
                                        {1, 1, 1, 1, 1},
                                        {0, 0, 0, 0, 0}};
        TVector<float> target{0, 1, 2, 1, 0};
        TVector<float> weight{1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeTotalF1Metric(3);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.229, 1e-3);
    }
}

}
