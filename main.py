import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl


@dataclass
class SelectedStats:
    """A container for the calculated statistics."""

    count: int
    min: Optional[float]
    max: Optional[float]
    sum: Optional[float]
    mean: Optional[float]


def process_csv(file_path: Path, column_name: str) -> SelectedStats:
    """
    Reads a CSV file and calculates descriptive statistics for a specified column using LazyFrame.

    This function uses the Polars lazy API to build an optimized query plan,
    which is ideal for performance on large datasets.
    """
    # Create a LazyFrame from the CSV file. This does not read the file yet,
    # only sets up the plan.
    try:
        lf = pl.scan_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file: {e}")

    # Build a query plan to calculate all statistics in a single pass.
    # We cast the target column to Float64 to ensure numeric operations are valid.
    aggregations = [
        pl.col(column_name).count().alias("count"),
        pl.col(column_name).cast(pl.Float64).min().alias("min"),
        pl.col(column_name).cast(pl.Float64).max().alias("max"),
        pl.col(column_name).cast(pl.Float64).sum().alias("sum"),
        pl.col(column_name).cast(pl.Float64).mean().alias("mean"),
    ]

    try:
        # Execute the query. This materializes the result into a DataFrame.
        # The resulting DataFrame will have a single row with our calculated stats.
        stats_df = lf.select(aggregations).collect()
    except pl.exceptions.ColumnNotFoundError:
        raise ValueError(f"Error: Column '{column_name}' not found in the file.")
    except Exception as e:
        raise RuntimeError(f"Error processing data with Polars: {e}")

    # Extract the results. The DataFrame has only one row.
    # Using .row(0, named=True) is a convenient way to get a dictionary of the results.
    stats_dict = stats_df.row(0, named=True)

    # Create and return a SelectedStats object.
    return SelectedStats(
        count=stats_dict["count"],
        min=stats_dict["min"],
        max=stats_dict["max"],
        sum=stats_dict["sum"],
        mean=stats_dict["mean"],
    )


def main():
    """Main function to parse arguments and print statistics."""
    parser = argparse.ArgumentParser(
        description="A CLI tool to calculate statistics for a numeric column in a CSV file.",
        epilog="Example: python main.py -f data/test.csv -c 'Amount Received'",
    )
    parser.add_argument(
        "-f",
        "--file-path",
        type=Path,
        required=True,
        help="The path to the CSV file.",
    )
    parser.add_argument(
        "-c",
        "--column-name",
        type=str,
        default="Amount Received",
        help="The name of the column to analyze.",
    )
    args = parser.parse_args()

    try:
        # Execute the data processing function.
        stats = process_csv(args.file_path, args.column_name)

        # Helper to format Option<float> values consistently to 4 decimal places.
        def format_opt(val: Optional[float]) -> str:
            return f"{val:.4f}" if val is not None else "N/A"

        # Print the results line by line.
        print("Output for python-polars")
        print(f"--- Statistics for '{args.column_name}' ---")
        print(f"Count: {stats.count}")
        print(f"Min:   {format_opt(stats.min)}")
        print(f"Max:   {format_opt(stats.max)}")
        print(f"Sum:   {format_opt(stats.sum)}")
        print(f"Mean:  {format_opt(stats.mean)}")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
