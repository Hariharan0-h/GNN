"""
Runtime Decision Engine

Action Labels
-------------
0 = NO_ACTION
1 = THREAD_ADJUST
2 = REBALANCE
"""


class RuntimeDecisionEngine:

    NO_ACTION = 0
    THREAD_ADJUST = 1
    REBALANCE = 2

    @staticmethod
    def predict(feature):

        degree_skew = abs(
            float(feature[0])
        )

        imbalance = float(
            feature[1]
        )

        clustering = float(
            feature[2]
        )

        density = float(
            feature[3]
        )

        score = (
            0.40 * degree_skew +
            0.30 * imbalance +
            0.20 * density +
            0.10 * clustering
        )

        # Strong structural imbalance
        if score >= 0.75:

            return (
                RuntimeDecisionEngine.REBALANCE
            )

        # Moderate imbalance
        elif score >= 0.45:

            return (
                RuntimeDecisionEngine.THREAD_ADJUST
            )

        # Stable partition
        else:

            return (
                RuntimeDecisionEngine.NO_ACTION
            )