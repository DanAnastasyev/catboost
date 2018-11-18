#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <library/unittest/registar.h>

Y_UNIT_TEST_SUITE(AccuracyMetricTest) {

Y_UNIT_TEST(BinaryAccuracyTest) {
    {
        TVector <TVector<double>> approx{{0, 1, 0, 0, 1, 0}};
        TVector<float> target{0, 1, 0, 0, 0, 1};
        TVector<float> weight{1, 1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeAccuracyMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.667, 1e-3);
    }
    {
        TVector <TVector<double>> approx{{0, 0, 1}};
        TVector<float> target{0, 1, 1};
        TVector<float> weight{1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeAccuracyMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);;

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.667, 1e-2);
    }
    {
        TVector <TVector<double>> approx{{1, 1, 1, 0}};
        TVector<float> target{1, 1, 1, 0};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeAccuracyMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
    }
    {
        TVector <TVector<double>> approx{{1, 1, 1, 1}};
        TVector<float> target{1, 1, 1, 1};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeAccuracyMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
    }
    {
        TVector <TVector<double>> approx{{0, 0, 0, 0}};
        TVector<float> target{0, 0, 0, 0};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeAccuracyMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
    }

}

Y_UNIT_TEST(MulticlassAccuracyTest) {
    {
        TVector<TVector<double>> approx{{1, 0, 0},
                                        {1, 0, 0},
                                        {0, 0, 1}};
        TVector<float> target{0, 1, 2};
        TVector<float> weight{1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeAccuracyMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.667, 1e-3);
    }
    {
        TVector<TVector<double>> approx{{0, 1, 0},
                                        {1, 0, 0},
                                        {0, 1, 0}};
        TVector<float> target{0, 1, 2};
        TVector<float> weight{1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeAccuracyMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0, 1e-1);
    }
    {
        TVector<TVector<double>> approx{{1, 0, 0},
                                        {0, 1, 0},
                                        {0, 0, 1}};
        TVector<float> target{0, 1, 2};
        TVector<float> weight{1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeAccuracyMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
    }
}

}
