#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <library/unittest/registar.h>

Y_UNIT_TEST_SUITE(MCCTest) {

Y_UNIT_TEST(MCCTest) {
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0, 0},
                                        {1, 1, 0, 0, 0},
                                        {0, 0, 1, 1, 1}};
        TVector<float> target{0, 1, 2, 1, 0};
        TVector<float> weight{1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeMCCMetric(3);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.217, 1e-3);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0, 0},
                                        {1, 1, 0, 1, 0},
                                        {0, 0, 1, 0, 1}};
        TVector<float> target{0, 1, 2, 1, 0};
        TVector<float> weight{1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeMCCMetric(3);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.505, 1e-3);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0, 0},
                                        {1, 1, 1, 1, 1},
                                        {0, 0, 0, 0, 0}};
        TVector<float> target{0, 1, 2, 1, 0};
        TVector<float> weight{1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeMCCMetric(3);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0., 1e-3);
    }
}

}
