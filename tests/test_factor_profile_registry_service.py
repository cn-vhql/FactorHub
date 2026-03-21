"""
Unit tests for FactorProfileRegistryService
Covers requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9
"""
import pytest
from backend.services.factor_profile_registry_service import FactorProfileRegistryService


@pytest.fixture
def service():
    return FactorProfileRegistryService()


# ---------------------------------------------------------------------------
# 1.2 load_profiles filters by mode
# ---------------------------------------------------------------------------

class TestLoadProfiles:
    def test_load_temporal_pool_profiles_returns_only_temporal_pool(self, service):
        profiles = service.load_profiles("temporal_pool")
        assert len(profiles) > 0
        for p in profiles:
            assert p["mode"] == "temporal_pool"

    def test_load_cross_sectional_profiles_returns_only_cross_sectional(self, service):
        profiles = service.load_profiles("cross_sectional_filter")
        assert len(profiles) > 0
        for p in profiles:
            assert p["mode"] == "cross_sectional_filter"

    def test_load_profiles_unknown_mode_returns_empty(self, service):
        profiles = service.load_profiles("nonexistent_mode")
        assert profiles == []

    def test_load_profiles_temporal_pool_contains_known_ids(self, service):
        profiles = service.load_profiles("temporal_pool")
        ids = [p["id"] for p in profiles]
        assert "momentum_v1" in ids

    def test_load_profiles_cross_sectional_contains_known_ids(self, service):
        profiles = service.load_profiles("cross_sectional_filter")
        ids = [p["id"] for p in profiles]
        assert "cross_sectional_momentum_v1" in ids


# ---------------------------------------------------------------------------
# 1.3 get_profile returns correct profile when id exists
# ---------------------------------------------------------------------------

class TestGetProfile:
    def test_get_existing_temporal_profile(self, service):
        profile = service.get_profile("momentum_v1")
        assert profile["id"] == "momentum_v1"
        assert profile["mode"] == "temporal_pool"

    def test_get_existing_cross_sectional_profile(self, service):
        profile = service.get_profile("cross_sectional_momentum_v1")
        assert profile["id"] == "cross_sectional_momentum_v1"
        assert profile["mode"] == "cross_sectional_filter"

    def test_get_profile_returns_full_config(self, service):
        profile = service.get_profile("momentum_v1")
        assert "factors" in profile
        assert "params" in profile
        assert len(profile["factors"]) > 0

    # 1.4 KeyError when profile_id not found
    def test_get_profile_raises_key_error_for_missing_id(self, service):
        with pytest.raises(KeyError):
            service.get_profile("does_not_exist")

    def test_get_profile_key_error_message_contains_id(self, service):
        missing_id = "totally_missing_profile"
        with pytest.raises(KeyError, match=missing_id):
            service.get_profile(missing_id)


# ---------------------------------------------------------------------------
# 1.5 validate_profile: missing required fields
# ---------------------------------------------------------------------------

class TestValidateProfileMissingFields:
    def _valid_profile(self):
        return {
            "id": "test_v1",
            "mode": "temporal_pool",
            "factors": [{"name": "momentum_20", "direction": 1, "weight": 1.0, "role": "signal"}],
            "params": {"top_n": 10},
        }

    def test_valid_profile_passes(self, service):
        service.validate_profile(self._valid_profile())  # should not raise

    def test_missing_id_raises_value_error(self, service):
        p = self._valid_profile()
        del p["id"]
        with pytest.raises(ValueError, match="id"):
            service.validate_profile(p)

    def test_missing_mode_raises_value_error(self, service):
        p = self._valid_profile()
        del p["mode"]
        with pytest.raises(ValueError, match="mode"):
            service.validate_profile(p)

    def test_missing_factors_raises_value_error(self, service):
        p = self._valid_profile()
        del p["factors"]
        with pytest.raises(ValueError, match="factors"):
            service.validate_profile(p)

    def test_missing_params_raises_value_error(self, service):
        p = self._valid_profile()
        del p["params"]
        with pytest.raises(ValueError, match="params"):
            service.validate_profile(p)

    def test_missing_multiple_fields_raises_value_error(self, service):
        p = {"mode": "temporal_pool"}
        with pytest.raises(ValueError):
            service.validate_profile(p)


# ---------------------------------------------------------------------------
# 1.6 validate_profile: signal weight sum != 1
# ---------------------------------------------------------------------------

class TestValidateProfileWeightSum:
    def _profile_with_weights(self, weights):
        factors = [
            {"name": f"f{i}", "direction": 1, "weight": w, "role": "signal"}
            for i, w in enumerate(weights)
        ]
        return {
            "id": "test_v1",
            "mode": "temporal_pool",
            "factors": factors,
            "params": {},
        }

    def test_weight_sum_exactly_one_passes(self, service):
        p = self._profile_with_weights([0.5, 0.5])
        service.validate_profile(p)  # should not raise

    def test_weight_sum_within_tolerance_passes(self, service):
        # 0.999 is the lower bound
        p = self._profile_with_weights([0.5, 0.499])
        service.validate_profile(p)

    def test_weight_sum_too_low_raises_value_error(self, service):
        p = self._profile_with_weights([0.3, 0.3])  # sum = 0.6
        with pytest.raises(ValueError):
            service.validate_profile(p)

    def test_weight_sum_too_high_raises_value_error(self, service):
        p = self._profile_with_weights([0.7, 0.7])  # sum = 1.4
        with pytest.raises(ValueError):
            service.validate_profile(p)

    def test_risk_filter_factors_excluded_from_weight_check(self, service):
        # risk_filter factors with weight=0 should not affect signal weight sum
        p = {
            "id": "test_v1",
            "mode": "temporal_pool",
            "factors": [
                {"name": "signal_f", "direction": 1, "weight": 1.0, "role": "signal"},
                {"name": "risk_f", "direction": -1, "weight": 0.0, "role": "risk_filter"},
            ],
            "params": {},
        }
        service.validate_profile(p)  # should not raise

    def test_no_signal_factors_skips_weight_check(self, service):
        # If there are no signal factors, weight check is skipped
        p = {
            "id": "test_v1",
            "mode": "temporal_pool",
            "factors": [
                {"name": "risk_f", "direction": -1, "weight": 0.0, "role": "risk_filter"},
            ],
            "params": {},
        }
        service.validate_profile(p)  # should not raise


# ---------------------------------------------------------------------------
# 1.7 validate_profile: invalid mode
# ---------------------------------------------------------------------------

class TestValidateProfileMode:
    def _base_profile(self, mode):
        return {
            "id": "test_v1",
            "mode": mode,
            "factors": [{"name": "f1", "direction": 1, "weight": 1.0, "role": "signal"}],
            "params": {},
        }

    def test_temporal_pool_mode_passes(self, service):
        service.validate_profile(self._base_profile("temporal_pool"))

    def test_cross_sectional_filter_mode_passes(self, service):
        service.validate_profile(self._base_profile("cross_sectional_filter"))

    def test_invalid_mode_raises_value_error(self, service):
        with pytest.raises(ValueError):
            service.validate_profile(self._base_profile("invalid_mode"))

    def test_empty_mode_raises_value_error(self, service):
        with pytest.raises(ValueError):
            service.validate_profile(self._base_profile(""))

    def test_mode_error_message_contains_mode_value(self, service):
        bad_mode = "bad_mode_xyz"
        with pytest.raises(ValueError, match=bad_mode):
            service.validate_profile(self._base_profile(bad_mode))


# ---------------------------------------------------------------------------
# 1.8 validate_profile: top_n out of range
# ---------------------------------------------------------------------------

class TestValidateProfileTopN:
    def _profile_with_top_n(self, top_n):
        return {
            "id": "test_v1",
            "mode": "temporal_pool",
            "factors": [{"name": "f1", "direction": 1, "weight": 1.0, "role": "signal"}],
            "params": {"top_n": top_n},
        }

    def test_top_n_1_passes(self, service):
        service.validate_profile(self._profile_with_top_n(1))

    def test_top_n_100_passes(self, service):
        service.validate_profile(self._profile_with_top_n(100))

    def test_top_n_50_passes(self, service):
        service.validate_profile(self._profile_with_top_n(50))

    def test_top_n_0_raises_value_error(self, service):
        with pytest.raises(ValueError):
            service.validate_profile(self._profile_with_top_n(0))

    def test_top_n_101_raises_value_error(self, service):
        with pytest.raises(ValueError):
            service.validate_profile(self._profile_with_top_n(101))

    def test_top_n_negative_raises_value_error(self, service):
        with pytest.raises(ValueError):
            service.validate_profile(self._profile_with_top_n(-1))

    def test_no_top_n_in_params_passes(self, service):
        # top_n is optional; absence should not raise
        p = {
            "id": "test_v1",
            "mode": "temporal_pool",
            "factors": [{"name": "f1", "direction": 1, "weight": 1.0, "role": "signal"}],
            "params": {},
        }
        service.validate_profile(p)


# ---------------------------------------------------------------------------
# 1.9 validate_profile: percentile_window out of range
# ---------------------------------------------------------------------------

class TestValidateProfilePercentileWindow:
    def _profile_with_pw(self, pw):
        return {
            "id": "test_v1",
            "mode": "temporal_pool",
            "factors": [{"name": "f1", "direction": 1, "weight": 1.0, "role": "signal"}],
            "params": {"percentile_window": pw},
        }

    def test_percentile_window_20_passes(self, service):
        service.validate_profile(self._profile_with_pw(20))

    def test_percentile_window_504_passes(self, service):
        service.validate_profile(self._profile_with_pw(504))

    def test_percentile_window_252_passes(self, service):
        service.validate_profile(self._profile_with_pw(252))

    def test_percentile_window_19_raises_value_error(self, service):
        with pytest.raises(ValueError):
            service.validate_profile(self._profile_with_pw(19))

    def test_percentile_window_505_raises_value_error(self, service):
        with pytest.raises(ValueError):
            service.validate_profile(self._profile_with_pw(505))

    def test_percentile_window_0_raises_value_error(self, service):
        with pytest.raises(ValueError):
            service.validate_profile(self._profile_with_pw(0))

    def test_no_percentile_window_in_params_passes(self, service):
        # percentile_window is optional; absence should not raise
        p = {
            "id": "test_v1",
            "mode": "temporal_pool",
            "factors": [{"name": "f1", "direction": 1, "weight": 1.0, "role": "signal"}],
            "params": {},
        }
        service.validate_profile(p)
