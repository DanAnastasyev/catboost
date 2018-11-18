#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <library/unittest/registar.h>

Y_UNIT_TEST_SUITE(PrecisionRecallMetricsTest) {

Y_UNIT_TEST(BinaryPrecisionTest) {
    {
        TVector<TVector<double>> approx{{0, 1, 0, 0, 1, 0}};
        TVector<float> target{0, 1, 0, 0, 0, 1};
        TVector<float> weight{1, 1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeBinClassPrecisionMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1. / (1. + 1.), 1e-2);
    }
    {
        TVector<TVector<double>> approx{{1, 1, 1, 1, 1, 1}};
        TVector<float> target{0, 1, 0, 0, 0, 1};
        TVector<float> weight{1, 1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeBinClassPrecisionMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 2. / (2. + 4.), 1e-3);
    }
    {
        TVector<TVector<double>> approx{{1, 1, 1, 1, 1, 1}};
        TVector<float> target{1, 1, 1, 1, 1, 1};
        TVector<float> weight{1, 1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeBinClassPrecisionMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 6. / (6. + 0.), 1e-3);
    }
}

Y_UNIT_TEST(BinaryRecallTest) {
    {
        TVector<TVector<double>> approx{{0, 1, 0, 0, 1, 0}};
        TVector<float> target{0, 1, 0, 0, 0, 1};
        TVector<float> weight{1, 1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeBinClassRecallMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1. / (1. + 1.), 1e-2);
    }
    {
        TVector<TVector<double>> approx{{1, 1, 1, 1, 1, 1}};
        TVector<float> target{0, 1, 0, 0, 0, 1};
        TVector<float> weight{1, 1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeBinClassRecallMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 2. / (2. + 0.), 1e-3);
    }
    {
        TVector<TVector<double>> approx{{1, 1, 1, 1, 1, 1}};
        TVector<float> target{1, 1, 1, 1, 1, 1};
        TVector<float> weight{1, 1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeBinClassRecallMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 6. / (6. + 0.), 1e-3);
    }
}

Y_UNIT_TEST(MulticlassPrecisionTest) {
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0, 0},
                                        {1, 1, 0, 0, 0},
                                        {0, 0, 1, 1, 1}};
        TVector<float> target{0, 1, 2, 1, 0};
        TVector<float> weight{1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeMultiClassPrecisionMetric(1);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1. / (1. + 1.), 1e-2);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0, 0},
                                        {1, 1, 0, 1, 0},
                                        {0, 0, 1, 0, 1}};
        TVector<float> target{0, 1, 2, 1, 0};
        TVector<float> weight{1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeMultiClassPrecisionMetric(1);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 2. / (2. + 1.), 1e-1);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0, 0},
                                        {1, 1, 1, 1, 1},
                                        {0, 0, 0, 0, 0}};
        TVector<float> target{0, 1, 2, 1, 0};
        TVector<float> weight{1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeMultiClassPrecisionMetric(1);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 2. / (2. + 4.), 1e-1);
    }
}

Y_UNIT_TEST(MulticlassRecallTest) {
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0, 0},
                                        {1, 1, 0, 0, 0},
                                        {0, 0, 1, 1, 1}};
        TVector<float> target{0, 1, 2, 1, 0};
        TVector<float> weight{1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeMultiClassRecallMetric(1);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1. / (1. + 1.), 1e-2);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0, 0},
                                        {1, 1, 0, 1, 0},
                                        {0, 0, 1, 0, 1}};
        TVector<float> target{0, 1, 2, 1, 0};
        TVector<float> weight{1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeMultiClassRecallMetric(1);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1., 1e-1);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0, 0},
                                        {1, 1, 1, 1, 1},
                                        {0, 0, 0, 0, 0}};
        TVector<float> target{0, 1, 2, 1, 0};
        TVector<float> weight{1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = MakeMultiClassRecallMetric(1);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1., 1e-1);
    }
}

}
