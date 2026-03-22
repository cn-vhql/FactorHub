"""
CLI script for the temporal filter validation workflow.

Usage:
    python scripts/run_temporal_filter_validation.py \
        --pool-csv scripts/data/pool100.csv \
        --start-date 2024-01-01 \
        --end-date 2026-03-20
"""
import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run temporal filter validation workflow"
    )
    parser.add_argument("--pool-csv", default="scripts/data/pool100.csv", help="Path to stock pool CSV")
    parser.add_argument("--start-date", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-03-21", help="End date YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=10, help="Top N stocks to select")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Score threshold for strategy D",
    )
    parser.add_argument(
        "--rebalance", default="W-FRI", help="Rebalance frequency (default: W-FRI)"
    )
    parser.add_argument(
        "--horizons",
        default="5,10,20",
        help="Comma-separated horizons (default: 5,10,20)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Thread pool size (default: 4)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/reports/temporal_filter_validation/",
        help="Output directory",
    )
    parser.add_argument(
        "--n-quantiles", type=int, default=5, help="Number of quantile groups (default: 5)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    horizons_tuple = tuple(int(x) for x in args.horizons.split(","))

    try:
        from backend.services.temporal_scoring_service import temporal_scoring_service
        from backend.services.temporal_filter_validation_service import (
            TemporalFilterValidationService,
        )

        # Step 1: Refresh factor expressions
        temporal_scoring_service.refresh_factor_expressions(force=True)

        # Step 2: Instantiate validation service, override output dir if provided
        validation_service = TemporalFilterValidationService()
        if args.output_dir != "data/reports/temporal_filter_validation/":
            validation_service.output_base_dir = Path(args.output_dir)
            validation_service.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Step 3: Load stock pool
        codes = validation_service.load_fixed_pool(args.pool_csv)

        # Step 4: Build score panel
        score_panel = validation_service.build_score_panel(
            codes, args.start_date, args.end_date, max_workers=args.max_workers
        )

        # Step 5: Build forward returns
        forward_returns = validation_service.build_forward_returns(
            codes, args.start_date, args.end_date, horizons=horizons_tuple
        )

        # Short-circuit on coverage-gate failure
        if isinstance(forward_returns, dict) and forward_returns.get("result_status") == "invalid_run":
            reason_codes = forward_returns.get("reason_codes", [])
            coverage = forward_returns.get("code_coverage", 0.0)
            print(
                f"invalid_run: 数据覆盖率门禁触发，reason_codes={reason_codes}，"
                f"code_coverage={coverage:.2%}",
                file=sys.stderr,
            )
            sys.exit(2)

        # Step 6: Build research panel
        research_panel = validation_service.build_research_panel(
            score_panel, forward_returns
        )

        # Step 7: Calculate cross-sectional IC
        ic_results = validation_service.calculate_cross_sectional_ic(
            research_panel, horizons=horizons_tuple
        )

        # Step 8: Analyze quantile spread
        quantile_results = validation_service.analyze_quantile_spread(
            research_panel, n_quantiles=args.n_quantiles, horizons=horizons_tuple
        )

        # Step 9: Run portfolio backtest
        portfolio_curves = validation_service.run_portfolio_backtest(
            score_panel,
            top_n=args.top_n,
            rebalance=args.rebalance,
            score_threshold=args.score_threshold,
        )

        # Step 10: Compare strategies
        strategy_comparison = validation_service.compare_strategies(portfolio_curves)

        # Step 11: Generate run ID and save results
        run_id = str(uuid.uuid4())
        out_dir = validation_service.save_results(
            run_id,
            score_panel,
            forward_returns,
            ic_results,
            quantile_results,
            portfolio_curves,
            strategy_comparison,
            temporal_scoring_service.factor_expression_source,
        )

        print(str(out_dir))

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
