"""Local metadata coverage, never raw data verification or automatic registration."""
from __future__ import annotations

SOURCES = {
    "proposal": "configs/genetic_primus_forward_research_proposal_v1.json",
    "research_contract": "configs/genetic_primus_research_execution_contract_v1.json",
    "release": "configs/genetic_primus_release_candidate_v1.json",
    "latency": "configs/genetic_primus_linear_latency_sensitivity_v1.json",
    "linear_semantics": "configs/genetic_primus_rolling_linear_semantics_v2.json",
    "ohlcv_manifest": "Reports/Exia/Genetic_Primus/prospective_3d_v1_202609/data_manifest_origin_09.json",
    "recorded_manifest": "Reports/Exia/Genetic_Primus/recorded_admission_v1_20260911/package_manifest.json",
    "ticker_integrity": "Reports/Exia/Genetic_Primus/recorded_archives_v1_20260914/ticker_integrity.json",
    "ticker_coverage": "Reports/Exia/Genetic_Primus/recorded_archives_v1_20260914/ticker_coverage_001.json",
    "l2_coverage": "Reports/Exia/Genetic_Primus/recorded_archives_v1_20260914/l2_coverage_009.json",
    "feature_definition": "Reports/Exia/Genetic_Primus/asof_features_v1_20261007/kernel_definition.json",
    "execution_source": "src/exia/genetic_primus/execution_replay_v1.py",
    "accounting_source": "src/exia/genetic_primus/accounting_v2.py",
}


def legacy_cost_evidence(values):
    """Aggregate round-trip scenarios do not identify fee or slippage per side."""
    import math
    if type(values) is not list or not 1 <= len(values) <= 8:
        raise ValueError("bounded list required")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 10_000 for v in values):
        raise ValueError("finite nonnegative scenario budgets required")
    return {"source_values": list(values), "source_unit": "aggregate_bps_per_round_trip",
            "target_unit": "separate_fee_and_slippage_bps_per_side",
            "per_side_fee_identified": False, "per_side_slippage_identified": False,
            "auto_mapping_allowed": False}


def pointer(document, locator):
    if type(locator) is not str or not locator.startswith("/"):
        raise ValueError("JSON pointer required")
    value = document
    for part in locator[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if type(value) is not dict or key not in value:
            raise ValueError("missing inspected metadata field: " + locator)
        value = value[key]
    return value


def coverage(materials, identities, api):
    api["keys"](materials, SOURCES)
    api["keys"](identities, SOURCES)
    for role in SOURCES:
        api["keys"](identities[role], ("bytes", "sha256"))
        api["integer"](identities[role]["bytes"], 1)
        api["digest"](identities[role]["sha256"])
        expected = str if role.endswith("_source") else dict
        if type(materials[role]) is not expected:
            raise ValueError("source type mismatch: " + role)
    contract = materials["research_contract"]
    assessment = api["assess"](contract)
    if len(assessment["missing_registration_fields"]) != 21 or assessment["registration_complete"] is not False:
        raise ValueError("this immutable audit is scoped to the 21-field unregistered draft")
    if api["sha"](materials["proposal"]) != api["PROPOSAL_SHA"]:
        raise ValueError("proposal changed")
    latency = materials["latency"]
    if latency["source_manifest_path"] != SOURCES["ohlcv_manifest"]:
        raise ValueError("legacy source manifest path substitution")
    if latency["source_manifest_sha256"] != identities["ohlcv_manifest"]["sha256"]:
        raise ValueError("legacy source manifest hash mismatch")
    # This is a pinned baseline audit, not acceptance of arbitrary future claims.
    if materials["recorded_manifest"]["evidence_kind"] != "synthetic":
        raise ValueError("recorded admission baseline changed")
    required_false = {
        "ticker_integrity": ("source_authenticity_verified", "coverage_verified", "qualification_eligible"),
        "ticker_coverage": ("content_semantics_verified", "funding_settlements_verified", "qualification_eligible"),
        "l2_coverage": ("complete", "content_semantics_verified", "funding_settlements_verified", "qualification_eligible"),
    }
    for role, fields in required_false.items():
        if any(materials[role][field] is not False for field in fields):
            raise ValueError("changed baseline claims require a separate audit: " + role)
    for key in ("instrument_rules", "account_fee_schedule", "book_mark_funding_dataset", "latency_slippage_calibration"):
        if materials["release"]["evidence"][key] is not None:
            raise ValueError("release evidence baseline changed")
    if latency["no_model_refits"] is not True:
        raise ValueError("latency baseline changed")
    costs = legacy_cost_evidence(latency["round_trip_cost_bps"])

    def ref(role, locator, unit="not_numeric"):
        if role.endswith("_source"):
            if type(locator) is not str or locator not in materials[role]:
                raise ValueError("missing static source anchor")
            value = locator
            kind = "static_code_anchor_not_execution_evidence"
        elif locator == "@file_sha256":
            value = identities[role]["sha256"]
            kind = "metadata_file_byte_hash_not_underlying_data_attestation"
        else:
            value = pointer(materials[role], locator)
            kind = "archived_metadata_claim_not_independent_verification"
        return {"source_role": role, "source_path": SOURCES[role], "source_file_sha256": identities[role]["sha256"],
                "locator": locator, "source_value": value, "source_unit": unit, "evidence_kind": kind}

    rows = []

    def row(field, status, references, reason, action, unit="SHA256"):
        if field not in assessment["missing_registration_fields"] or any(r["field"] == field for r in rows):
            raise ValueError("field mapping substitution/duplicate")
        rows.append({"field": field, "target_unit": unit, "evidence_status": status,
                     "references": references, "blocking_reason": reason, "required_action": action,
                     "registration_value": None, "auto_registration_allowed": False})

    row("timing.training_budget_ms", "missing_registration",
        [ref("proposal", "/clock/training_latency_budget_seconds", "seconds")],
        "Compute budget не выбран; duration реальных fits не измерялась в этом этапе.",
        "Зарегистрировать planning budget в ms и его основание; каждый будущий fit измерять отдельно.", "ms")
    row("timing.execution_delay_ms", "legacy_diagnostic_only",
        [ref("latency", "/delay_bars", "4h_bars"), ref("latency", "/interpretation")],
        "0/1 bar sensitivity не является измеренной latency; 1 bar также вне scope нового delay < bar.",
        "Явно выбрать исследовательскую delay assumption; не выдавать её за latency биржи.", "ms")
    row("timing.budget_evidence_sha256", "missing_registration",
        [ref("research_contract", "/timing/budget_evidence_sha256")],
        "Нет документа-основания timing budget; SHA текущего кода его не заменяет.",
        "Связать planning assumptions и единицы отдельным immutable документом.")
    row("data.manifest_sha256", "legacy_metadata_only",
        [ref("ohlcv_manifest", "@file_sha256"), ref("ohlcv_manifest", "/schema_version")],
        "Существующий manifest относится к старому frozen контракту, а не новой исследовательской выборке.",
        "Построить manifest выбранных inputs без чтения/раскрытия нового sealed interval; отдельно проверить данные.")
    row("data.timestamp_spec_sha256", "unverified_metadata_only",
        [ref("ohlcv_manifest", "/metadata/columns"), ref("ohlcv_manifest", "/metadata/timeframe_ms", "ms"),
         ref("execution_source", "`ts` is local receive time")],
        "Имена OHLCV columns и runtime receive-time правило не доказывают open/close/arrival semantics этих данных.",
        "Закрепить dataset-specific timestamp semantics и преобразования; проверить origin grid и timezone.")
    row("data.availability_evidence_sha256", "unverified_metadata_only",
        [ref("ohlcv_manifest", "/metadata/columns"), ref("ticker_coverage", "/local_utc_range"),
         ref("ticker_coverage", "/content_semantics_verified")],
        "В OHLCV manifest нет arrival columns; BTC ticker одного дня не подтверждает availability всего train/full8.",
        "Разделить наблюдаемый arrival и зарегистрированную OHLCV availability assumption; не смешивать доказательства.")
    row("data.revision_policy_sha256", "missing_registration",
        [ref("research_contract", "/policies/missing_or_revised_data")],
        "Fail-closed правило есть, dataset-specific политика версий и исправлений не зарегистрирована.",
        "Описать immutable revisions, получение corrections и доступную as-of версию.")
    row("data.universe_sha256", "legacy_metadata_only",
        [ref("ohlcv_manifest", "/metadata/symbols"), ref("release", "/symbol_map")],
        "Старый full8 и futures symbol map существуют, но новый universe/instrument basis не выбран.",
        "Явно зарегистрировать universe с instrument/venue/settlement идентичностью.")
    row("data.feature_definition_sha256", "engineering_definition_only",
        [ref("feature_definition", "@file_sha256"), ref("feature_definition", "/return_12"),
         ref("feature_definition", "/efficiency_12"), ref("feature_definition", "/features_available_at")],
        "Формулы реализованы и проверены на synthetic; declared decision-asof не является measured compute availability.",
        "Версионировать исследовательское определение без synthetic-attestation подмены; привязать реальный prefix/availability.")
    row("data.training_prefix_manifest_sha256", "missing_registration",
        [ref("research_contract", "/policies/training_identity"), ref("ohlcv_manifest", "/metadata/coverage_end_exclusive", "epoch_ms")],
        "Legacy full archive SHA не равен training-only prefix SHA и включает post-train данные.",
        "Фиксировать per-origin prefix и revisions до training-asof, отдельно от deploy provenance.")
    row("data.snapshot_cutoff_ms", "legacy_metadata_only",
        [ref("ohlcv_manifest", "/metadata/coverage_end_exclusive", "epoch_ms")],
        "Coverage end старого архива не является availability cutoff новой выборки.",
        "Зарегистрировать cutoff новой snapshot с timestamp/availability semantics.", "epoch_ms")
    row("costs.adapter_sha256", "incompatible_adapter",
        [ref("execution_source", "Funding is supplied at settlement time"),
         ref("release", "/execution_semantics/fill_rule"), ref("release", "/execution_semantics/ohlcv_fill_fallback")],
        "Snapshot IOC adapter и OHLCV open sensitivity имеют разные fill semantics; code SHA не является проверкой fills.",
        "Отдельно реализовать research accounting adapter; для release сохранить snapshot IOC без OHLCV fallback.")
    for field in ("normal_fee_bps_per_side", "normal_slippage_bps_per_side",
                  "stress_fee_bps_per_side", "stress_slippage_bps_per_side"):
        row("costs." + field, "legacy_budget_only",
            [ref("latency", "/round_trip_cost_bps", "aggregate_bps_per_round_trip"),
             ref("release", "/evidence/account_fee_schedule")],
            "16/24/40 bps — aggregate сценарии. Деление на 2 не определяет отдельно fee и slippage.",
            "Зафиксировать fee schedule и slippage/stress assumptions per side с независимым основанием.", "bps_per_side")
    row("costs.calibration_evidence_sha256", "legacy_diagnostic_only",
        [ref("latency", "/interpretation"), ref("latency", "/no_model_refits"),
         ref("release", "/evidence/latency_slippage_calibration")],
        "Sensitivity без refit не измеряет latency/slippage и не является calibration реального исполнения.",
        "Разделить preregistered sensitivity assumptions и отдельную execution calibration.")
    row("costs.funding_evidence_sha256", "unverified_metadata_only",
        [ref("ticker_coverage", "/funding_timestamps", "reported_microseconds"),
         ref("ticker_coverage", "/funding_settlements_verified"), ref("ticker_integrity", "/source_authenticity_verified")],
        "Наличие funding timestamps/rates не подтверждает settlements и authenticity; покрывается BTC за один день.",
        "Подтвердить semantics actual settlements и полное покрытие требуемых instruments/дат; иначе gate закрыт.")
    row("costs.borrow_basis_sha256", "missing_registration",
        [ref("release", "/execution_target")],
        "Linear perpetual в старом release не выбирает instrument нового research; applicability borrow не зарегистрирована.",
        "Документировать применимость/N/A borrow отдельно от funding и margin costs.")
    row("costs.terminal_policy_sha256", "engineering_definition_only",
        [ref("accounting_source", "if close_time == end_timestamp:"), ref("accounting_source", '"interval_end"'),
         ref("research_contract", "/policies/terminal")],
        "Cash accounting реализует terminal close по close; новая terminal timestamp/price availability ещё не привязана.",
        "Зарегистрировать единственный global-end close и fees/funding, не вводить forced close на каждом retrain.")
    row("costs.instrument_spec_sha256", "missing_registration",
        [ref("release", "/execution_target"), ref("release", "/evidence/instrument_rules")],
        "Target Bitget USDT futures объявлен, но instrument rules evidence = null.",
        "Получить versioned локальные specs tick/lot/min notional/contract/margin/settlement; сеть автоматически не открывать.")
    if set(r["field"] for r in rows) != set(assessment["missing_registration_fields"]):
        raise ValueError("incomplete field coverage")
    counts = {}
    for r in rows:
        counts[r["evidence_status"]] = counts.get(r["evidence_status"], 0) + 1
    return {"schema": "exia.genetic_primus.registration_coverage/1",
            "scope": "curated_local_sources_not_exhaustive_repo_or_raw_data_verification",
            "contract_sha256": assessment["contract_sha256"], "source_count": len(SOURCES),
            "rows": sorted(rows, key=lambda r: r["field"]), "status_counts": counts,
            "registered_fields": 0, "remaining_registration_fields": 21,
            "legacy_cost_interpretation": costs,
            "observations": {
                "legacy_ohlcv_coverage": materials["ohlcv_manifest"]["metadata"],
                "recorded_admission_kind": materials["recorded_manifest"]["evidence_kind"],
                "ticker_complete_claim": materials["ticker_coverage"]["complete"],
                "ticker_rows_claim": materials["ticker_coverage"]["rows"],
                "ticker_crc_verified_claim": materials["ticker_integrity"]["gzip_crc_eof_verified"],
                "ticker_authenticity_verified_claim": materials["ticker_integrity"]["source_authenticity_verified"],
                "funding_settlements_verified_claim": materials["ticker_coverage"]["funding_settlements_verified"],
                "last_inspected_l2_complete_claim": materials["l2_coverage"]["complete"],
                "last_inspected_l2_exchange_range": materials["l2_coverage"]["exchange_utc_range"],
            },
            "raw_market_data_read": False, "actual_provenance_verified": False,
            "auto_registration_allowed": False, "fit_allowed": False,
            "market_research_launch_ready": False, "real_money_ready": False,
            "safety": dict(contract["safety"])}
