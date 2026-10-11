"""
analyzer.py - Core quality checks for the Dataset Quality Checker.
Takes a pandas DataFrame and returns a plain dictionary report,
so Flask (or any other interface) can display it.
"""
import json
import sys

import numpy as np
import pandas as pd


def format_size(n_bytes):
    """Show bytes as B, KB or MB depending on size."""
    n_bytes = int(n_bytes)
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes / 1024 ** 2:.2f} MB"


def basic_info(df):
    numeric = df.select_dtypes(include=np.number).shape[1]
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": int(numeric),
        "other_columns": int(df.shape[1] - numeric),
        "memory": format_size(df.memory_usage(deep=True).sum()),
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
    }


def column_profile(df):
    """One row per column: type, missing, unique values and a few examples."""
    profile = []
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        examples = [str(v) for v in s.dropna().unique()[:3]]
        profile.append({
            "column": str(col),
            "dtype": str(s.dtype),
            "missing": n_missing,
            "missing_pct": round(n_missing / len(df) * 100, 2) if len(df) else 0.0,
            "unique": int(s.nunique(dropna=True)),
            "examples": examples,
        })
    return profile


def missing_values(df):
    counts = df.isna().sum()
    result = []
    for col in df.columns:
        n = int(counts[col])
        if n > 0:
            result.append({
                "column": str(col),
                "missing": n,
                "percent": round(n / len(df) * 100, 2),
            })
    return sorted(result, key=lambda r: r["percent"], reverse=True)


def duplicate_rows(df):
    n = int(df.duplicated().sum())
    pct = round(n / len(df) * 100, 2) if len(df) else 0.0
    return {"count": n, "percent": pct}


def constant_columns(df, threshold=0.99):
    """Columns with a single value, or where one value fills >= 99% of rows."""
    result = []
    for col in df.columns:
        vc = df[col].value_counts(dropna=False)
        if len(vc) == 0:
            continue
        top_share = vc.iloc[0] / len(df)
        if len(vc) == 1:
            result.append({"column": str(col), "type": "constant", "top_share": 100.0})
        elif top_share >= threshold:
            result.append({"column": str(col), "type": "near-constant",
                           "top_share": round(top_share * 100, 2)})
    return result


def _numeric_as_float(df):
    """Numeric columns as float arrays (handles nullable dtypes safely)."""
    numeric = df.select_dtypes(include=np.number)
    return {str(col): numeric[col].to_numpy(dtype=float, na_value=np.nan)
            for col in numeric.columns}


def infinite_values(df):
    result = []
    for col, values in _numeric_as_float(df).items():
        n = int(np.isinf(values).sum())
        if n:
            result.append({"column": col, "count": n})
    return result


def outliers_iqr(df):
    """Outliers per numeric column using the 1.5 * IQR rule."""
    result = []
    for col, values in _numeric_as_float(df).items():
        s = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
        if s.nunique() < 2:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < low) | (s > high)).sum())
        if n:
            result.append({"column": col, "count": n,
                           "percent": round(n / len(s) * 100, 2),
                           "low": round(float(low), 4), "high": round(float(high), 4)})
    return sorted(result, key=lambda r: r["percent"], reverse=True)


def class_balance(df, label_col):
    if not label_col or label_col not in df.columns:
        return None
    counts = df[label_col].value_counts(dropna=False)
    total = int(counts.sum())
    classes = [{"label": "(missing)" if pd.isna(k) else str(k), "count": int(v),
                "percent": round(v / total * 100, 2)} for k, v in counts.items()]
    smallest = counts.iloc[-1]
    ratio = float(counts.iloc[0] / smallest) if smallest else float("inf")
    return {"label_column": label_col, "classes": classes,
            "imbalance_ratio": round(ratio, 2)}


def high_correlations(df, threshold=0.95, limit=50):
    numeric = pd.DataFrame(_numeric_as_float(df)).replace([np.inf, -np.inf], np.nan)
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr().abs()
    cols = list(corr.columns)
    result = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iat[i, j]
            if pd.notna(v) and v >= threshold:
                result.append({"column_a": cols[i], "column_b": cols[j],
                               "correlation": round(float(v), 3)})
    result.sort(key=lambda r: r["correlation"], reverse=True)
    return result[:limit]


def quality_score(report):
    """Simple 0-100 score. Each problem type has a maximum penalty."""
    rows = report["basic"]["rows"] or 1
    cols = report["basic"]["columns"] or 1
    score = 100.0

    missing_cells = sum(m["missing"] for m in report["missing"])
    score -= min(30, missing_cells / (rows * cols) * 100 * 2)
    score -= min(15, report["duplicates"]["percent"])
    score -= min(15, len(report["constant"]) / cols * 100)
    score -= min(15, len(report["infinite"]) * 3)
    score -= min(10, len(report["correlations"]))

    cb = report["class_balance"]
    if cb:
        if cb["imbalance_ratio"] > 10:
            score -= 15
        elif cb["imbalance_ratio"] > 3:
            score -= 7

    return max(0, round(score))


def build_checks(report):
    """Turn the raw results into a validation checklist with pass / warn / fail status."""
    rows = report["basic"]["rows"] or 1
    cols = report["basic"]["columns"] or 1
    checks = []

    # Dataset size (informational - does not change the score)
    n = report["basic"]["rows"]
    checks.append({
        "id": "size", "name": "Dataset size",
        "status": "pass" if n >= 30 else "warn",
        "summary": f"{n:,} rows - enough for reliable checks." if n >= 30 else
                   f"Only {n} row(s). Results may be unreliable on such a small dataset.",
    })

    # Missing values
    missing_cells = sum(m["missing"] for m in report["missing"])
    missing_pct = round(missing_cells / (rows * cols) * 100, 2)
    worst = report["missing"][0]["percent"] if report["missing"] else 0
    checks.append({
        "id": "missing", "name": "Missing values",
        "status": "pass" if missing_cells == 0 else ("fail" if worst > 30 else "warn"),
        "summary": ("No missing values." if missing_cells == 0 else
                    f"{missing_cells} empty cells ({missing_pct}% of the dataset) "
                    f"in {len(report['missing'])} column(s)."),
    })

    # Duplicates
    d = report["duplicates"]
    checks.append({
        "id": "duplicates", "name": "Duplicate rows",
        "status": "pass" if d["count"] == 0 else ("fail" if d["percent"] > 10 else "warn"),
        "summary": "No duplicate rows." if d["count"] == 0 else
                   f"{d['count']} duplicate rows ({d['percent']}%).",
    })

    # Constant columns
    c = report["constant"]
    checks.append({
        "id": "constant", "name": "Constant columns",
        "status": "pass" if not c else "warn",
        "summary": "Every column carries information." if not c else
                   f"{len(c)} column(s) have (almost) a single value.",
    })

    # Infinite values
    inf = report["infinite"]
    total_inf = sum(i["count"] for i in inf)
    checks.append({
        "id": "infinite", "name": "Infinite values",
        "status": "pass" if not inf else "fail",
        "summary": "No infinite values." if not inf else
                   f"{total_inf} infinite value(s) in {len(inf)} column(s). Most ML models will crash on these.",
    })

    # Outliers (informational - outliers are often real attacks in IDS data)
    o = report["outliers"]
    checks.append({
        "id": "outliers", "name": "Outliers (1.5 × IQR)",
        "status": "pass" if not o else "warn",
        "summary": "No outliers detected." if not o else
                   f"{len(o)} column(s) contain outliers. Review them: in intrusion data they are often the attacks.",
    })

    # Class balance
    cb = report["class_balance"]
    if cb is None:
        checks.append({
            "id": "class_balance", "name": "Class balance", "status": "skip",
            "summary": "Not checked. Choose a label column to include it (it also affects the score).",
        })
    else:
        r = cb["imbalance_ratio"]
        checks.append({
            "id": "class_balance", "name": "Class balance",
            "status": "pass" if r <= 3 else ("fail" if r > 10 else "warn"),
            "summary": f"{len(cb['classes'])} classes, largest-to-smallest ratio {r}:1.",
        })

    # Correlations
    corr = report["correlations"]
    checks.append({
        "id": "correlations", "name": "Highly correlated features",
        "status": "pass" if not corr else "warn",
        "summary": "No redundant feature pairs (|r| ≥ 0.95)." if not corr else
                   f"{len(corr)} feature pair(s) with |r| ≥ 0.95. These features are likely redundant.",
    })

    return checks


def recommendations(report):
    recs = []
    for m in report["missing"]:
        if m["percent"] > 50:
            recs.append(f"Column '{m['column']}' is {m['percent']}% missing - consider dropping it.")
        else:
            recs.append(f"Column '{m['column']}' has {m['percent']}% missing values - impute (mean/median/mode) or drop those rows.")
    if report["duplicates"]["count"]:
        recs.append(f"Remove {report['duplicates']['count']} duplicate rows to avoid biased training and data leakage.")
    for c in report["constant"]:
        recs.append(f"Column '{c['column']}' is {c['type']} - it adds little information; consider dropping it.")
    for inf in report["infinite"]:
        recs.append(f"Column '{inf['column']}' has {inf['count']} infinite values - replace with NaN, then impute or drop.")
    if report["correlations"]:
        recs.append(f"{len(report['correlations'])} highly correlated feature pairs found - consider removing one feature from each pair.")
    cb = report["class_balance"]
    if cb and cb["imbalance_ratio"] > 3:
        recs.append(f"Classes are imbalanced (ratio {cb['imbalance_ratio']}:1) - consider SMOTE, class weights, or stratified sampling.")
    if not recs:
        recs.append("No major issues found. The dataset looks clean.")
    return recs


def analyze(df, label_col=None):
    report = {
        "basic": basic_info(df),
        "profile": column_profile(df),
        "missing": missing_values(df),
        "duplicates": duplicate_rows(df),
        "constant": constant_columns(df),
        "infinite": infinite_values(df),
        "outliers": outliers_iqr(df),
        "class_balance": class_balance(df, label_col),
        "correlations": high_correlations(df),
    }
    report["score"] = quality_score(report)
    report["checks"] = build_checks(report)
    report["recommendations"] = recommendations(report)
    return report


if __name__ == "__main__":
    # Usage: python analyzer.py data.csv [label_column]
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <file.csv> [label_column]")
        sys.exit(1)
    data = pd.read_csv(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(analyze(data, label), indent=2))