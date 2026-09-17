"""Generate a LaTeX LP table (requires booktabs, makecell, multirow, graphicx, siunitx).

The CSV's *_num_0 columns are interpreted as NONZERO counts. Color counts
are divided by vocabulary_size - 266. Bias groups share one table with multirow bias cells.
The document must define \\usedcolor, \\usededgesfree, and \\usededgespriced.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys


COLUMNS = (
    "bias", "vocabulary_size", "lp_value", "num_steps", "time",
    "c_num_1", "c_num_0", "f_num_1", "f_num_0", "p_num_1", "p_num_0",
)
COUNT_COLUMNS = ("vocabulary_size", "num_steps", *COLUMNS[5:])


def read_rows(path: Path) -> list[dict[str, Decimal]]:
    rows = []
    errors = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        header = [cell.strip() for cell in next(reader, [])]
        if len(header) != len(set(header)) or set(header) != set(COLUMNS):
            raise ValueError("CSV must have exactly these columns: " + ", ".join(COLUMNS))
        for cells in reader:
            if not cells or all(not cell.strip() for cell in cells):
                continue
            line = reader.line_num
            if len(cells) != len(header):
                errors.append(f"line {line}: expected {len(header)} fields, found {len(cells)}")
                continue
            try:
                row = {key: Decimal(value.strip()) for key, value in zip(header, cells)}
                if any(not value.is_finite() or value < 0 for value in row.values()):
                    raise ValueError("values must be finite and nonnegative")
                if any(row[key] != row[key].to_integral_value() for key in COUNT_COLUMNS):
                    raise ValueError("sizes, steps, and counts must be integers")
                if row["vocabulary_size"] <= 266:
                    raise ValueError("vocabulary_size must be greater than 266")
            except (InvalidOperation, ValueError) as exc:
                errors.append(f"line {line}: invalid numeric data ({exc})")
                continue
            rows.append(row)
    if errors:
        raise ValueError("Invalid CSV:\n" + "\n".join(errors))
    if not rows:
        raise ValueError("CSV contains no data rows")
    return rows


def render_table(rows: list[dict[str, Decimal]]) -> str:
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\caption{Characteristics of the solutions for the LP.",
        r"\\ \% of 1s (and of $\neg$ 0s) measures the ratio of $\usedcolor$ which are 1 (or not 0) divided by the LP vocabulary budget ($\text{vocabulary size}-266$).}",
        r"\label{tab:lp_running_metrics}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{Bias}",
        r"& \multirow{2}{*}{\makecell{Vocabulary \\ Size}}",
        r"& & & & \multicolumn{2}{c}{$\usedcolor$}",
        r"& \multicolumn{2}{c}{$\usededgesfree$}",
        r"& \multicolumn{2}{c}{$\usededgespriced$} \\",
        r"\cmidrule(lr){6-7}",
        r"\cmidrule(lr){8-9}",
        r"\cmidrule(lr){10-11}",
        r"& & \makecell{\# steps} & \makecell{Time (sec)} & \makecell{LP Value}",
        r"& \makecell{\% of 1s} & \makecell{\% of $\neg$ 0s}",
        r"& \makecell{\# of 1s} & \makecell{\# of $\neg$ 0s}",
        r"& \makecell{\# of 1s} & \makecell{\# of $\neg$ 0s} \\",
        r"\midrule",
    ]
    for group_index, bias in enumerate(sorted({row["bias"] for row in rows})):
        group = sorted(
            (row for row in rows if row["bias"] == bias),
            key=lambda row: row["vocabulary_size"],
        )
        bias_text = format(bias.normalize(), "f")
        if group_index:
            lines.append(r"\midrule")
        for row_index, row in enumerate(group):
            size = int(row["vocabulary_size"])
            budget = Decimal(size - 266)
            vocabulary = f"${size // 1024}k$" if size % 1024 == 0 else rf"\num{{{size}}}"
            bias_cell = (
                rf"\multirow{{{len(group)}}}{{*}}{{${bias_text}$}}"
                if row_index == 0 else ""
            )
            cells = [
                bias_cell,
                vocabulary,
                rf"\num{{{row['num_steps']:.0f}}}",
                f"{row['time']:.3f}",
                rf"\num{{{row['lp_value']:f}}}",
                rf"${100 * row['c_num_1'] / budget:.2f}\%$",
                rf"${100 * row['c_num_0'] / budget:.2f}\%$",
                *(rf"\num{{{row[key]:.0f}}}" for key in COLUMNS[7:]),
            ]
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule", r"\end{tabular}}", r"\vspace{-5pt}", r"\end{table}",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=Path("util_lp_value.csv"))
    parser.add_argument("-o", "--output", type=Path, help="Write LaTeX here (default: stdout)")
    parser.add_argument("--bias", type=Decimal, help="Only generate the table for this bias")
    args = parser.parse_args()
    try:
        rows = read_rows(args.csv_path)
        if args.bias is not None:
            rows = [row for row in rows if row["bias"] == args.bias]
            if not rows:
                raise ValueError(f"No rows found for bias {args.bias}")
        output = render_table(rows) + "\n"
        if args.output is None:
            sys.stdout.write(output)
        else:
            args.output.write_text(output, encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
