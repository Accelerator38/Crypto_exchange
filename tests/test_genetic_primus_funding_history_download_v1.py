"""Pure synthetic page-parser tests; never REST, collector or pytest discovery."""
from __future__ import annotations

import copy
import unittest
from exia.genetic_primus import funding_history_download_v1 as f


def payload(symbol="BTCUSDT", count=100, offset=0):
    return {"code": "00000", "requestTime": f.END_MS+24*f.HOUR_MS,
            "data": [{"symbol": symbol, "fundingRate": "0.0001",
                      "fundingTime": str(f.END_MS+16*f.HOUR_MS-(offset+i)*8*f.HOUR_MS)} for i in range(count)]}


def pages():
    return [{"page_no": 1, "payload": payload()}, {"page_no": 2, "payload": payload(offset=100)}]


class FundingTests(unittest.TestCase):
    def test_decimal_units_signed_and_zero(self):
        self.assertEqual(f.decimal_rate("0.0001"), ("0.0001", "1"))
        self.assertEqual(f.decimal_rate("-0.00005"), ("-0.00005", "-0.5"))
        self.assertEqual(f.decimal_rate("-0.000"), ("0", "0"))

    def test_bad_or_unbounded_rates(self):
        for value in ("NaN", "Infinity", "1", "-1", "1e-999999", True, .001, "0."+"0"*100):
            with self.assertRaises(ValueError):
                f.decimal_rate(value)

    def test_timestamp_strict(self):
        for value in (True, 1.2, "-1", 0, 2**53, "1e12"):
            with self.assertRaises(ValueError):
                f.timestamp(value)

    def test_envelope_and_symbol(self):
        for update in ({"code": "429"}, {"data": {}}, {"requestTime": None}):
            with self.assertRaises(ValueError):
                f.parse_page(payload() | update, "BTCUSDT")
        with self.assertRaises(ValueError):
            f.parse_page(payload("ETHUSDT"), "BTCUSDT")

    def test_descending_unique_not_future(self):
        for modification in ("duplicate", "reverse", "future"):
            value = payload()
            if modification == "duplicate":
                value["data"][1] = value["data"][0]
            elif modification == "reverse":
                value["data"].reverse()
            else:
                value["data"][0]["fundingTime"] = str(value["requestTime"]+1)
            with self.assertRaises(ValueError):
                f.parse_page(value, "BTCUSDT")

    def test_page_bound_and_exact_fraction(self):
        value = f.parse_page(payload(), "BTCUSDT")
        self.assertEqual(value["row_count"], 100)
        self.assertEqual(value["rows"][0]["rate_bps"], "1")
        with self.assertRaises(ValueError):
            f.parse_page(payload(count=101), "BTCUSDT")

    def test_json_duplicate_nonfinite_and_byte_cap(self):
        for raw in (b'{"code":"00000","code":"bad"}', b'{"x":NaN}', b' '* (f.MAX_RESPONSE_BYTES+1)):
            with self.assertRaises(ValueError):
                f.strict_json(raw)

    def test_interval_boundary_and_rows(self):
        result = f.audit_symbol(pages(), "BTCUSDT")
        self.assertTrue(result["api_enumeration_ready"])
        self.assertEqual(result["requested_rows"], 183)
        self.assertEqual(result["observed_gap_hours"], {"8.0": 183})
        self.assertFalse(result["financial_replay_ready"])
        self.assertFalse(result["settlement_schedule_independently_verified"])

    def test_no_zero_fallback(self):
        result = f.audit_symbol([{"page_no": 1, "payload": payload(count=0)}], "BTCUSDT")
        self.assertEqual(result["events"], [])
        self.assertFalse(result["api_enumeration_ready"])
        self.assertFalse(result["more_pages_useful"])

    def test_one_page_needs_older_pages(self):
        result = f.audit_symbol(pages()[:1], "BTCUSDT")
        self.assertTrue(result["more_pages_useful"])
        self.assertFalse(result["boundary_bracketed"])

    def test_identical_overlap_deduplicated(self):
        bank = pages()
        bank[1]["payload"] = payload(offset=99)
        result = f.audit_symbol(bank, "BTCUSDT")
        self.assertEqual(result["duplicates_equal_rate"], 1)
        self.assertTrue(result["api_enumeration_ready"])

    def test_conflicting_overlap_rejected(self):
        bank = pages()
        bank[1]["payload"] = payload(offset=99)
        bank[1]["payload"]["data"][0]["fundingRate"] = "0.0002"
        with self.assertRaises(ValueError):
            f.audit_symbol(bank, "BTCUSDT")

    def test_nonprogress_or_noncontiguous_pages(self):
        for bank in ([pages()[0], copy.deepcopy(pages()[0])], [pages()[1]]):
            if len(bank) == 2:
                bank[1]["page_no"] = 2
            with self.assertRaises(ValueError):
                f.audit_symbol(bank, "BTCUSDT")

    def test_missing_settlement_is_not_imputed(self):
        bank = pages()
        bank[1]["payload"]["data"].pop(30)
        result = f.audit_symbol(bank, "BTCUSDT")
        self.assertIn("unexplained_observed_gap", result["issues"])
        self.assertFalse(result["api_enumeration_ready"])

    def test_provider_before_period_not_future_evidence(self):
        value = payload()
        delta = 100*24*f.HOUR_MS
        value["requestTime"] -= delta
        for row in value["data"]:
            row["fundingTime"] = str(int(row["fundingTime"])-delta)
        result = f.audit_symbol([{"page_no": 1, "payload": value}], "BTCUSDT")
        self.assertTrue(result["server_latest_predates_requested_interval"])
        self.assertFalse(result["api_enumeration_ready"])


if __name__ == "__main__":
    unittest.main()
