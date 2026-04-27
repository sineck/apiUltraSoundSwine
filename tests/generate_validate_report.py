from __future__ import annotations

from run_validate_compare import build_report, write_report_files, REPORT_HTML


def main() -> int:
    report = build_report()
    write_report_files(report)
    print(str(REPORT_HTML))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
